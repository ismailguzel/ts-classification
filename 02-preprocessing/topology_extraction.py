"""
Topological Feature Extraction for Time Series (Persistent Homology)

Mirrors the interface of feature_extraction.py so output feeds directly into
feature_selection.py and train.py.

Each persistence diagram is vectorised into 8 features:
    5 Carlsson coordinates (f1–f5) + persistent entropy + landscape L1 + landscape L2

--method controls which diagrams are computed:

  sublevel+h1  (default)
      sub_H0   : sublevel set H0 — captures mean shifts, valley structure
      sup_H0   : superlevel set H0 — captures point anomalies, peaks
      H1       : Takens + Vietoris-Rips H1 — captures periodicity, loops
      Total    : 3 × 8 = 24 features   (requires Ripser for H1)

  sublevel
      sub_H0 + sup_H0 only — no Ripser needed
      Total: 2 × 8 = 16 features

  takens
      Takens H0 + H1 via Vietoris-Rips (Ripser)
      Total: 2 × 8 = 16 features

  both
      sub_H0 + sup_H0 + H0 + H1 — all diagrams
      Total: 4 × 8 = 32 features

Usage:
    # Default (sublevel+h1):
    python topology_extraction.py \\
        --input  ../data/raw/dataset \\
        --output ../data/features/dataset/topological \\
        --n-jobs -1

    # Sublevel only (fastest, no Ripser):
    python topology_extraction.py \\
        --input ../data/raw/dataset \\
        --output ../data/features/dataset/topological \\
        --method sublevel

    # Takens with auto-selected embedding parameters:
    python topology_extraction.py \\
        --method takens --auto-delay --auto-dim --n-perm 500

    # Shape rather than amplitude (mandatory for the season-* modes):
    python topology_extraction.py --zscore
"""

import argparse
import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")

REQUIRED_COLUMNS = ["series_id", "time", "data", "is_stationary",
                    "primary_category", "sub_category"]


# ---------------------------------------------------------------------------
# Diagram statistics
# ---------------------------------------------------------------------------

def carlsson_coordinates(diagram: np.ndarray) -> np.ndarray:
    """5 Carlsson coordinates for a persistence diagram.

    As defined in Carlsson et al. (see also Umeda 2017):
        p_i     = d_i - b_i          (lifetime / persistence)
        d_max   = max(d_i)

        f1 = Σ  b_i · p_i
        f2 = Σ  (d_max - d_i) · p_i
        f3 = Σ  b_i² · p_i⁴
        f4 = Σ  (d_max - d_i)² · p_i⁴
        f5 = max p_i                 (most persistent bar)

    These are provably stable under small Wasserstein perturbations of the
    diagram.  f1 encodes birth-weighted persistence (trend/level structure);
    f2 encodes how far deaths are from the maximum filtration value; f3/f4
    are higher-degree versions that emphasise long-lived bars; f5 captures
    the single most prominent topological feature.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]

    if len(diag) == 0:
        return np.zeros(5)

    b, d  = diag[:, 0], diag[:, 1]
    p     = d - b
    d_max = float(d.max())

    return np.array([
        np.sum(b * p),                    # f1
        np.sum((d_max - d) * p),          # f2
        np.sum(b ** 2 * p ** 4),          # f3
        np.sum((d_max - d) ** 2 * p ** 4),# f4
        float(p.max()),                   # f5
    ])


def persistent_entropy(diagram: np.ndarray) -> float:
    """Shannon entropy of normalised bar lifetimes: H = -Σ (pᵢ/L) log(pᵢ/L).

    Low entropy  → few dominant bars (e.g. one long bar from a mean shift).
    High entropy → many bars of similar length (e.g. white noise).
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]
    if len(diag) == 0:
        return 0.0
    p = diag[:, 1] - diag[:, 0]
    L = p.sum()
    if L < 1e-12:
        return 0.0
    probs = p / L
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _persistence_landscape_vector(
    diagram: np.ndarray,
    n_landscapes: int,
    resolution: int,
    use_mean: bool,
) -> np.ndarray:
    """Mean persistence landscape discretised into `resolution` bins.

    Tent function per bar: λ(x) = min(x−b, d−x) if b ≤ x ≤ d, else 0.
    Top-n_landscapes values stored per grid point; averaged if use_mean=True.
    """
    diag = diagram[np.isfinite(diagram).all(axis=1)]
    diag = diag[diag[:, 1] > diag[:, 0]]

    out_len = resolution if use_mean else n_landscapes * resolution
    if len(diag) == 0:
        return np.zeros(out_len)

    min_b, max_d = diag[:, 0].min(), diag[:, 1].max()
    if max_d <= min_b:
        return np.zeros(out_len)

    grid = np.linspace(min_b, max_d, resolution)
    lands = np.zeros((n_landscapes, resolution))
    births, deaths = diag[:, 0], diag[:, 1]

    for i, x in enumerate(grid):
        active = (births <= x) & (x <= deaths)
        if active.any():
            vals = np.sort(np.minimum(x - births[active], deaths[active] - x))[::-1]
            k = min(n_landscapes, len(vals))
            lands[:k, i] = vals[:k]

    return lands.mean(axis=0) if use_mean else lands.flatten()


def diagram_to_stats(diagram: np.ndarray,
                     n_landscapes: int = 5,
                     resolution: int = 100) -> np.ndarray:
    """Compact 8-feature summary for one persistence diagram.

    [carlsson×5, persistent_entropy, landscape_L1_norm, landscape_L2_norm]
    """
    carl    = carlsson_coordinates(diagram)
    entropy = persistent_entropy(diagram)
    lvec    = _persistence_landscape_vector(diagram, n_landscapes, resolution, use_mean=True)
    l1      = float(np.sum(np.abs(lvec)))
    l2      = float(np.sqrt(np.sum(lvec ** 2)))
    return np.concatenate([carl, [entropy, l1, l2]])


def diagram_to_vector(diagram: np.ndarray,
                      mode: str,
                      n_landscapes: int = 5,
                      resolution: int = 100,
                      use_mean: bool = True) -> np.ndarray:
    """Vectorise one persistence diagram. Only mode='stats' is supported."""
    return diagram_to_stats(diagram, n_landscapes, resolution)


def _diagram_feature_len(mode: str, resolution: int) -> int:
    return len(_STAT_SUFFIXES)  # 8: 5 Carlsson + entropy + L1 + L2


# ---------------------------------------------------------------------------
# Sublevel / superlevel set persistence (union-find, no Ripser)
# ---------------------------------------------------------------------------

def sublevel_persistence_1d(ts: np.ndarray) -> np.ndarray:
    """H0 persistence via sublevel set filtration on a 1D path complex.

    Models the time series as f: {0,...,n-1} → R.  Union-find with the
    elder rule: the younger component (higher min = born later) dies when it
    merges with an older one.

    Returns array of shape (n_pairs, 2) with columns [birth, death].
    The essential class (global minimum) is omitted.

    Useful for: mean shifts (long low-level bar), valley structure, trends.
    """
    n = len(ts)
    if n < 2:
        return np.zeros((0, 2))

    parent      = list(range(n))
    comp_min    = ts.tolist()   # min function value per component root

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    edges = sorted((max(ts[i], ts[i + 1]), i, i + 1) for i in range(n - 1))

    pairs: list[tuple[float, float]] = []
    for fval, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        bi, bj = comp_min[ri], comp_min[rj]
        if bi >= bj:           # ri is younger
            pairs.append((bi, fval))
            parent[ri] = rj
        else:                  # rj is younger
            pairs.append((bj, fval))
            parent[rj] = ri

    return np.array(pairs, dtype=float) if pairs else np.zeros((0, 2))


def superlevel_persistence_1d(ts: np.ndarray) -> np.ndarray:
    """H0 persistence via superlevel set filtration.

    Computed as sublevel persistence of -ts with signs flipped back.
    Returns (birth, death) pairs where birth ≥ death (we re-orient to
    birth < death before landscape computation).

    Useful for: point anomalies (isolated spike → very long isolated bar),
    peaks, and collective anomalies above the mean.
    """
    neg_pairs = sublevel_persistence_1d(-ts)
    if len(neg_pairs) == 0:
        return np.zeros((0, 2))
    # In negated space: birth_neg = -actual_death, death_neg = -actual_birth
    result = np.column_stack([-neg_pairs[:, 1], -neg_pairs[:, 0]])
    return result


# ---------------------------------------------------------------------------
# Takens parameter estimation
# ---------------------------------------------------------------------------

def estimate_delay_mi(ts: np.ndarray, max_lag: int = 50) -> int:
    """Select Takens delay τ via Mutual Information (Fraser & Swinney 1986)."""
    ts = np.asarray(ts, dtype=float)
    n_bins = max(10, int(np.sqrt(len(ts))))
    mi_vals = []
    for lag in range(1, max_lag + 1):
        x, y = ts[:-lag], ts[lag:]
        h, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = h / h.sum()
        px  = pxy.sum(axis=1, keepdims=True)
        py  = pxy.sum(axis=0, keepdims=True)
        mask = pxy > 0
        mi_vals.append(float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask]))))
    mi_arr = np.array(mi_vals)
    for i in range(1, len(mi_arr) - 1):
        if mi_arr[i] < mi_arr[i - 1] and mi_arr[i] < mi_arr[i + 1]:
            return i + 1
    return int(np.argmin(mi_arr)) + 1


def estimate_dimension_fnn(ts: np.ndarray, delay: int,
                            max_dim: int = 10,
                            r_tol: float = 15.0,
                            a_tol: float = 2.0,
                            threshold: float = 0.10) -> int:
    """Select Takens dimension via False Nearest Neighbors (Kennel et al. 1992)."""
    from sklearn.neighbors import NearestNeighbors
    ts    = np.asarray(ts, dtype=float)
    std   = ts.std() + 1e-10
    for d in range(1, max_dim + 1):
        n = len(ts) - (d - 1) * delay
        if n < 10:
            return d
        cloud = np.empty((n, d))
        for i in range(d):
            cloud[:, i] = ts[i * delay: i * delay + n]
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(cloud)
        dists, idxs = nbrs.kneighbors(cloud)
        nn_d   = dists[:, 1]
        nn_idx = idxs[:, 1]
        n2 = len(ts) - d * delay
        if n2 < 10:
            return d
        extra_s = ts[d * delay: d * delay + n2]
        extra_n = np.array([
            ts[nn_idx[i] + d * delay] if (nn_idx[i] + d * delay) < len(ts) else 0.0
            for i in range(n2)
        ])
        nn_d2 = nn_d[:n2] + 1e-10
        delta = np.abs(extra_s - extra_n)
        fnn   = ((delta / nn_d2 > r_tol) | (np.sqrt(nn_d2**2 + delta**2) / std > a_tol)).mean()
        if fnn < threshold:
            return d
    return max_dim


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

class SublevelFeatureExtractor:
    """Persistent homology via sublevel/superlevel set filtration.

    Diagrams: sub_H0 (valleys/mean-shifts) + sup_H0 (peaks/point-anomalies).
    No Takens embedding, no Ripser — pure NumPy.

    diagram_features:
        'stats'     → 2 × 11 = 22 features  (Carlsson + entropy + L1 + L2)
        'landscape' → 2 × resolution features (default: 200)
        'both'      → 2 × (11 + resolution) features (default: 222)
    """

    def __init__(self,
                 n_landscapes: int = 5,
                 landscape_resolution: int = 100,
                 normalize: bool = True):
        self.n_landscapes        = n_landscapes
        self.landscape_resolution = landscape_resolution
        self.normalize           = normalize
        self._feat_len           = _diagram_feature_len('stats', landscape_resolution)

    def extract_one(self, ts: np.ndarray) -> tuple[np.ndarray, None, None]:
        ts  = np.asarray(ts, dtype=float)
        sub = sublevel_persistence_1d(ts)
        sup = superlevel_persistence_1d(ts)
        v_sub = diagram_to_vector(sub, 'stats', self.n_landscapes, self.landscape_resolution)
        v_sup = diagram_to_vector(sup, 'stats', self.n_landscapes, self.landscape_resolution)
        vec = np.concatenate([v_sub, v_sup])
        if self.normalize and vec.max() > 0:
            vec = vec / vec.max()
        return vec, None, None

    def get_feature_names(self) -> list[str]:
        return _make_feature_names(
            ['sub_H0', 'sup_H0'],
            'stats',
            self.landscape_resolution,
        )

    def n_features(self) -> int:
        return 2 * self._feat_len


class TopologicalFeatureExtractor:
    """Persistent homology via Takens embedding + Vietoris-Rips (Ripser).

    Diagrams: H0 + H1 (configurable via --homology-dims).
    """

    def __init__(self,
                 embedding_dim: int = 3,
                 embedding_delay: int = 1,
                 auto_delay: bool = False,
                 auto_dim: bool = False,
                 max_delay: int = 50,
                 max_dim: int = 10,
                 fnn_threshold: float = 0.10,
                 homology_dims: tuple = (0, 1),
                 n_landscapes: int = 5,
                 landscape_resolution: int = 100,
                 normalize: bool = True,
                 max_thresh: float = np.inf,
                 n_perm: int | None = 500):
        try:
            from ripser import ripser as _r
            self._ripser = _r
        except ImportError:
            raise ImportError("ripser is not installed. Run: pip install ripser")
        self.embedding_dim        = embedding_dim
        self.embedding_delay      = embedding_delay
        self.auto_delay           = auto_delay
        self.auto_dim             = auto_dim
        self.max_delay            = max_delay
        self.max_dim              = max_dim
        self.fnn_threshold        = fnn_threshold
        self.homology_dims        = homology_dims
        self.n_landscapes         = n_landscapes
        self.landscape_resolution = landscape_resolution
        self.normalize            = normalize
        self.max_thresh           = max_thresh
        self.n_perm               = n_perm
        self._feat_len            = _diagram_feature_len('stats', landscape_resolution)

    def _resolve(self, ts):
        delay = estimate_delay_mi(ts, self.max_delay) if self.auto_delay else self.embedding_delay
        dim   = estimate_dimension_fnn(ts, delay, self.max_dim, threshold=self.fnn_threshold) \
                if self.auto_dim else self.embedding_dim
        return delay, dim

    def extract_one(self, ts: np.ndarray) -> tuple[np.ndarray, int, int]:
        delay, dim = self._resolve(ts)
        n     = len(ts) - (dim - 1) * delay
        if n < 2:
            raise ValueError(f"Series too short: len={len(ts)}, dim={dim}, delay={delay}")
        cloud = np.empty((n, dim))
        for i in range(dim):
            cloud[:, i] = ts[i * delay: i * delay + n]
        kwargs = {"maxdim": max(self.homology_dims), "thresh": self.max_thresh}
        if self.n_perm:
            kwargs["n_perm"] = min(self.n_perm, len(cloud))
        res  = self._ripser(cloud, **kwargs)
        dgms = {d: res["dgms"][d] if d < len(res["dgms"]) else np.zeros((0, 2))
                for d in self.homology_dims}
        vecs = [diagram_to_vector(dgms[d], 'stats',
                                  self.n_landscapes, self.landscape_resolution)
                for d in sorted(dgms)]
        vec = np.concatenate(vecs)
        if self.normalize and vec.max() > 0:
            vec /= vec.max()
        return vec, delay, dim

    def get_feature_names(self) -> list[str]:
        tags = [f"H{d}" for d in self.homology_dims]
        return _make_feature_names(tags, 'stats', self.landscape_resolution)

    def n_features(self) -> int:
        return len(self.homology_dims) * self._feat_len


class CombinedExtractor:
    """Runs Takens+Ripser and Sublevel extractors, concatenates features."""

    def __init__(self, takens: TopologicalFeatureExtractor,
                 sublevel: SublevelFeatureExtractor):
        self._takens   = takens
        self._sublevel = sublevel

    def extract_one(self, ts):
        ft, delay, dim = self._takens.extract_one(ts)
        fs, _, _       = self._sublevel.extract_one(ts)
        return np.concatenate([ft, fs]), delay, dim

    def get_feature_names(self):
        return self._takens.get_feature_names() + self._sublevel.get_feature_names()

    def n_features(self):
        return self._takens.n_features() + self._sublevel.n_features()


# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------

_STAT_SUFFIXES = [
    'carl_f1', 'carl_f2', 'carl_f3', 'carl_f4', 'carl_f5_max',
    'entropy', 'landscape_l1', 'landscape_l2',
]


def _make_feature_names(diagram_tags: list[str], mode: str, resolution: int) -> list[str]:
    names = []
    for tag in diagram_tags:
        for s in _STAT_SUFFIXES:
            names.append(f"topo__{tag}__{s}")
    return names


# ---------------------------------------------------------------------------
# Worker + Pipeline
# ---------------------------------------------------------------------------

def _zscore(ts: np.ndarray) -> np.ndarray:
    """Per-series standardisation: (x - mean) / std.

    Sub- and superlevel persistence is measured in the units of the signal, so
    without this the absolute amplitude of a series flows straight into its
    topological features. Whether that is wanted depends on the question:

      Study A  — keep raw scale. `variance_shift` and `volatility` are *about*
                 amplitude, so standardising away scale would remove the signal.
      Study B  — standardise. With betise 0.4.0 defaults, per-series sigma spans
                 ~0.2 (single_seasonality) to ~5.7 (sarima); a 25x gap that raw
                 amplitude alone would separate, telling us nothing about shape.

    Running both ways on the same data is also the only way to see how much of a
    diagram family's discriminative power is shape and how much is scale.
    """
    ts = np.asarray(ts, dtype=float)
    std = ts.std()
    return (ts - ts.mean()) / std if std > 1e-12 else ts - ts.mean()


def _extract_worker(series_id, values, extractor, zscore=False):
    try:
        if zscore:
            values = _zscore(values)
        feat, delay, dim = extractor.extract_one(values)
        return series_id, feat, None, delay, dim
    except Exception as exc:
        return series_id, None, str(exc), None, None


class TopologyPipeline:
    def __init__(self, extractor, n_jobs: int = -1, batch_size: int = 200,
                 zscore: bool = False):
        self.extractor  = extractor
        self.n_jobs     = n_jobs if n_jobs and n_jobs > 0 else os.cpu_count()
        self.batch_size = batch_size
        self.zscore     = zscore

    def _process_batch(self, batch):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_extract_worker)(sid, vals, self.extractor, self.zscore)
            for sid, vals in batch
        )
        ids, feats, errors, delays, dims = [], [], [], [], []
        for sid, feat, err, delay, dim in results:
            if err is None:
                ids.append(sid); feats.append(feat)
                delays.append(delay); dims.append(dim)
            else:
                errors.append((sid, err))
        return ids, feats, errors, delays, dims

    def run(self, input_path: Path, output_path: Path):
        files = sorted(input_path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files in {input_path}")
        print(f"Found {len(files)} parquet file(s).")

        df = pd.concat([pd.read_parquet(f, columns=REQUIRED_COLUMNS) for f in files],
                       ignore_index=True)
        initial = len(df)
        df = df.dropna(subset=["data"])
        df = df[~df["data"].isin([np.inf, -np.inf])]
        if len(df) < initial:
            print(f"Cleaned {initial - len(df)} bad rows.")

        labels = (df.groupby("series_id")[
            ["is_stationary", "primary_category", "sub_category"]
        ].first().reset_index())

        series_list = [(sid, grp["data"].values) for sid, grp in df.groupby("series_id")]
        n_total = len(series_list)

        print(f"Total series   : {n_total}")
        print(f"Features/series: {self.extractor.n_features()}")
        print(f"Parallel jobs  : {self.n_jobs}  |  Batch: {self.batch_size}")

        del df; gc.collect()

        all_ids, all_feats, all_errors = [], [], []
        all_delays, all_dims = [], []
        batches = [series_list[i: i + self.batch_size]
                   for i in range(0, n_total, self.batch_size)]

        for b_idx, batch in enumerate(tqdm(batches, desc="Batches", unit="batch"), 1):
            ids, feats, errors, delays, dims = self._process_batch(batch)
            all_ids.extend(ids); all_feats.extend(feats)
            all_errors.extend(errors)
            all_delays.extend(delays); all_dims.extend(dims)
            print(f"  Batch {b_idx}/{len(batches)}: {len(ids)} ok, {len(errors)} failed")
            gc.collect()

        if all_errors:
            print(f"\nWarning: {len(all_errors)} series failed.")
            for sid, err in all_errors[:5]:
                print(f"  series_id={sid}: {err}")

        feature_names = self.extractor.get_feature_names()
        feat_df = pd.DataFrame(all_feats, columns=feature_names)
        feat_df.insert(0, "series_id", all_ids)

        output_path.mkdir(parents=True, exist_ok=True)
        feat_df.to_parquet(output_path / "features.parquet", index=False)
        print(f"\nSaved features.parquet  {feat_df.shape}")

        with open(output_path / "feature_names.txt", "w") as fh:
            fh.write("\n".join(feature_names))

        lbl_out = labels[labels["series_id"].isin(all_ids)].reset_index(drop=True)
        lbl_out.to_parquet(output_path / "labels.parquet", index=False)
        print(f"Saved labels.parquet    {lbl_out.shape}")

        delays_arr = np.array([d for d in all_delays if d is not None])
        if len(delays_arr):
            dims_arr = np.array([d for d in all_dims if d is not None])
            print(f"\n  delay — mean={delays_arr.mean():.1f} "
                  f"median={np.median(delays_arr):.0f} "
                  f"min={delays_arr.min()} max={delays_arr.max()}")
            print(f"  dim   — mean={dims_arr.mean():.1f} "
                  f"median={np.median(dims_arr):.0f} "
                  f"min={dims_arr.min()} max={dims_arr.max()}")

        print(f"\nOutput: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Topological feature extraction for time series",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--method",
                   choices=["sublevel+h1", "sublevel", "takens", "both"],
                   default="sublevel+h1",
                   help=(
                       "sublevel+h1 : sub_H0 + sup_H0 + Takens H1 only — no overlap [default]\n"
                       "sublevel    : sub_H0 + sup_H0 (no Ripser)\n"
                       "takens      : Takens H0 + H1 (Ripser)\n"
                       "both        : sublevel + takens H0 + H1"
                   ))


    # Takens params
    p.add_argument("--embedding-dim",    type=int,   default=3)
    p.add_argument("--embedding-delay",  type=int,   default=1)
    p.add_argument("--auto-delay",       action="store_true")
    p.add_argument("--auto-dim",         action="store_true")
    p.add_argument("--max-delay",        type=int,   default=50)
    p.add_argument("--max-dim",          type=int,   default=10)
    p.add_argument("--fnn-threshold",    type=float, default=0.10)
    p.add_argument("--homology-dims",    type=str,   default="0,1")
    p.add_argument("--n-perm",           type=int,   default=500)
    p.add_argument("--max-thresh",       type=float, default=np.inf)

    # Landscape params (used internally for L1/L2 norm computation in stats)
    p.add_argument("--n-landscapes",         type=int, default=5)
    p.add_argument("--landscape-resolution", type=int, default=100)
    p.add_argument("--no-normalize",         action="store_true")
    p.add_argument("--zscore", action="store_true",
                   help="Standardise each series ((x-mean)/std) BEFORE the filtration and "
                        "the Takens embedding, so persistence measures shape rather than "
                        "amplitude. Required for the season-* modes, where per-series sigma "
                        "spans ~0.2 to ~5.7; leave off for shape, where variance_shift and "
                        "volatility are genuinely about scale.")

    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--n-jobs",     type=int, default=-1)

    args = p.parse_args()

    homology_dims = tuple(int(d) for d in args.homology_dims.split(","))
    n_perm   = args.n_perm if args.n_perm > 0 else None
    norm     = not args.no_normalize
    feat_per_diag = _diagram_feature_len('stats', args.landscape_resolution)

    common_kw = dict(
        n_landscapes=args.n_landscapes,
        landscape_resolution=args.landscape_resolution,
        normalize=norm,
    )
    takens_kw = dict(
        embedding_dim=args.embedding_dim,
        embedding_delay=args.embedding_delay,
        auto_delay=args.auto_delay,
        auto_dim=args.auto_dim,
        max_delay=args.max_delay,
        max_dim=args.max_dim,
        fnn_threshold=args.fnn_threshold,
        homology_dims=homology_dims,
        max_thresh=args.max_thresh,
        n_perm=n_perm,
        **common_kw,
    )

    if args.method == "sublevel+h1":
        # sub_H0 + sup_H0 (sublevel) + H1 only from Takens — no overlap, 3 diagrams
        extractor = CombinedExtractor(
            TopologicalFeatureExtractor(**{**takens_kw, "homology_dims": (1,)}),
            SublevelFeatureExtractor(**common_kw),
        )
        n_diags = 3
    elif args.method == "sublevel":
        extractor = SublevelFeatureExtractor(**common_kw)
        n_diags = 2
    elif args.method == "both":
        extractor = CombinedExtractor(
            TopologicalFeatureExtractor(**takens_kw),
            SublevelFeatureExtractor(**common_kw),
        )
        n_diags = 2 + len(homology_dims)
    else:  # takens
        extractor = TopologicalFeatureExtractor(**takens_kw)
        n_diags = len(homology_dims)

    print("=" * 70)
    print("Topological Feature Extraction")
    print("=" * 70)
    print(f"Filtration method  : {args.method}")
    print(f"Diagram features   : stats  ({feat_per_diag} per diagram)")
    print(f"Diagrams           : {n_diags}")
    print(f"Total features     : {extractor.n_features()}")
    print(f"Input              : {args.input}")
    print(f"Output             : {args.output}")
    if args.method in ("takens", "both", "sublevel+h1"):
        delay_str = f"auto (MI, max={args.max_delay})" if args.auto_delay \
                    else f"{args.embedding_delay} (fixed)"
        dim_str   = f"auto (FNN, max={args.max_dim})" if args.auto_dim \
                    else f"{args.embedding_dim} (fixed)"
        print(f"Takens delay       : {delay_str}")
        print(f"Takens dimension   : {dim_str}")
        print(f"n_perm             : {n_perm}")
    print(f"Per-series z-score : {'yes' if args.zscore else 'no (raw amplitude)'}")
    print(f"n_jobs             : {args.n_jobs}")
    print("=" * 70)

    TopologyPipeline(extractor, n_jobs=args.n_jobs, batch_size=args.batch_size,
                     zscore=args.zscore).run(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
