# Changelog

All notable changes to the Hierarchical Time Series Classification project.

---

## 2025-11-06 - Scalable Dataset Architecture

### Major Refactoring
- **Unified data generation**: Merged `generate_toy.py` into `generate.py` with `--scale` parameter
- **11 scale presets**: test (1.5K), 5k, 10k, 20k, 30k, 50k, 75k, 90k, 120k, 150k, 200k
- **Dynamic configuration**: `create_scaled_config()` function generates proportional configs from 90K baseline
- **SLURM integration**: Updated `full-data-job.sh` to accept scale as argument (`sbatch full-data-job.sh [scale]`)

### Enhanced
- **Config CLI**: `python config.py [scale]` now shows detailed breakdown with 5-class distribution
- **Documentation**: Complete rewrite of all README files to reflect scalable architecture
- **Strategic scaling guide**: Recommended progression for accuracy vs scale experiments

### Removed
- `generate_toy.py` (functionality moved to `generate.py --scale test`)
- Python cache files (`__pycache__/`)
- Backup files (`*.backup`)

### Migration Guide
**Old → New Commands:**
- `python generate_toy.py` → `python generate.py --scale test`
- `python generate.py` → `python generate.py --scale 90k`
- `sbatch full-data-job.sh` → `sbatch full-data-job.sh 90k`

**Benefits:**
- Single source of truth for data generation
- Easy scaling experiments (5K → 200K)
- Consistent proportions across all scales
- Simplified maintenance

---

## 2025-11-02 - ID integrity, features pipeline, Model 2 boost

### Added
- Global unique `series_id` at data generation via upstream `start_id` support; wired through generation scripts. Added `01-data-generation/verify_ids.py` to validate uniqueness across parquet files.
- Deterministic file ordering in `02-preprocessing/extract_features.py` using `sorted()` for reproducible `--max-files` behavior.
- Features-mode training enhancements:
	- Model 1: integrated XGBoost and CatBoost (optional) alongside RF/SVM.
	- Model 2: integrated CatBoost (optional) alongside RF/SVM/XGBoost.

### Changed
- Removed ad-hoc `series_id` remapping from feature extraction; uniqueness now enforced at source during generation.

### Notes
- CatBoost is optional (not pinned in requirements); when not installed, training scripts will skip it with a clear message.
- Use `02-preprocessing/feature_selection.py --target all` to produce both binary and primary selected feature sets consumed by Model 1 and Model 2 feature pipelines.

## 2025-11-01 - Minimal, focused updates

These updates keep the code clean and only address real needs observed in HPC deployments.

### Added
- `--n-jobs` CLI parameter to control training parallelism:
	- `03-models/hierarchical/model1_binary/train_model1.py`
	- `03-models/hierarchical/model2_nonstationary/train_model2.py`
- SLURM scripts now pass `--n-jobs ${SLURM_CPUS_PER_TASK}` so CPU allocation is respected:
	- `03-models/hierarchical/model1_binary/slurm_train_model1.sh`
	- `03-models/hierarchical/model2_nonstationary/slurm_train_model2.sh`

### Fixed
- Dependency compatibility pinned in `requirements.txt` to avoid known issues with ROCKET/Arsenal:
	- `numpy<2.0`, `scikit-learn<1.6`, `sktime<0.32`, `pandas<2.3`, `pyarrow<18`

### Notes
- No changes to data loading logic; existing PyArrow + ThreadPoolExecutor kept as-is.
- No broad refactor; focus is on stability and HPC flexibility only.
