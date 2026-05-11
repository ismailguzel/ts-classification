#!/usr/bin/env python3
"""
Baseline Comparison: Traditional Stationarity Tests

This script computes accuracy of traditional statistical tests:
- ADF (Augmented Dickey-Fuller) Test
- KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
- Phillips-Perron (PP) Test

Usage:
    python compute_baselines.py --data-path ../data/raw/unified-20k --output-dir results/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss
try:
    from arch.unitroot import PhillipsPerron
    PP_AVAILABLE = True
except ImportError:
    PP_AVAILABLE = False
    print("Warning: arch library not available, Phillips-Perron test will be skipped")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed


def load_test_data(data_path, sample_size=None):
    """
    Load test dataset from raw time series directories.
    
    Expected structure:
        data_path/
        ├── stationary/
        │   ├── ar/long.parquet  (columns: series_id, time, data, is_stationary, ...)
        │   ├── ma/long.parquet
        │   └── ...
        ├── deterministic_trend_linear/long.parquet
        ├── stochastic/long.parquet
        └── ...
    
    Each parquet file contains multiple series in long format (one row per time point).
    """
    data_path = Path(data_path)
    
    print(f"Loading raw time series from {data_path}...")
    
    all_series = []
    
    # Define patterns and their subdirectories
    # Stationary patterns: direct path
    stationary_patterns = [
        'stationary/ar',
        'stationary/ma', 
        'stationary/arma',
        'stationary/white_noise'
    ]
    
    # Non-stationary patterns: have subdirectories (up/down) + noise types
    nonstationary_base = [
        'deterministic_trend_linear',
        'deterministic_trend_quadratic',
        'deterministic_trend_cubic',
        'deterministic_trend_exponential',
        'deterministic_trend_damped',
        'stochastic',
        'volatility',
        'point_anomaly_single',
        'point_anomaly_multiple',
        'multi_collective_anomaly',
        'multi_mean_shift',
        'multi_variance_shift',
        'multi_trend_shift'
    ]
    
    subdirs = ['up', 'down']  # Some patterns have up/down
    noise_types = ['ar', 'ma', 'arma', 'white_noise']
    
    # Load stationary patterns
    for pattern in stationary_patterns:
        pattern_path = data_path / pattern / 'long.parquet'
        
        if not pattern_path.exists():
            print(f"  {pattern}: not found")
            continue
        
        print(f"  Loading {pattern}...")
        df = pd.read_parquet(pattern_path)
        is_stationary = 1
        
        # Group by series_id and convert to one row per series
        unique_series_ids = df['series_id'].unique()
        
        # Sample if needed
        if sample_size and len(unique_series_ids) > sample_size:
            unique_series_ids = np.random.choice(unique_series_ids, sample_size, replace=False)
        
        for series_id in unique_series_ids:
            series_data = df[df['series_id'] == series_id]
            ts_values = series_data.sort_values('time')['data'].values
            
            all_series.append({
                'series_id': f"{pattern}_{series_id}",
                'data': ts_values,
                'is_stationary': is_stationary,
                'pattern': pattern,
                'length': len(ts_values)
            })
        
        print(f"    {pattern}: {len(unique_series_ids)} series")
    
    # Load non-stationary patterns (nested structure - recursively find all long.parquet files)
    for base_pattern in nonstationary_base:
        base_path = data_path / base_pattern
        
        if not base_path.exists():
            print(f"  {base_pattern}: not found")
            continue
        
        print(f"  Loading {base_pattern}...")
        pattern_series_count = 0
        
        # Find all long.parquet files recursively
        parquet_files = list(base_path.rglob('long.parquet'))
        
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            is_stationary = 0
            
            # Get relative path for pattern name
            rel_path = parquet_file.relative_to(base_path).parent
            
            # Group by series_id
            unique_series_ids = df['series_id'].unique()
            
            # Sample if needed
            if sample_size and len(unique_series_ids) > sample_size:
                unique_series_ids = np.random.choice(unique_series_ids, sample_size, replace=False)
            
            for series_id in unique_series_ids:
                series_data = df[df['series_id'] == series_id]
                ts_values = series_data.sort_values('time')['data'].values
                
                all_series.append({
                    'series_id': f"{base_pattern}_{rel_path}_{series_id}",
                    'data': ts_values,
                    'is_stationary': is_stationary,
                    'pattern': f"{base_pattern}/{rel_path}",
                    'length': len(ts_values)
                })
            
            pattern_series_count += len(unique_series_ids)
        
        if pattern_series_count > 0:
            print(f"    {base_pattern}: {pattern_series_count} series")
    
    if not all_series:
        raise FileNotFoundError(f"No data files found in {data_path}")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(all_series)
    
    print(f"\nLoaded {len(result_df)} total series")
    print(f"  Stationary: {(result_df['is_stationary'] == 1).sum()}")
    print(f"  Non-stationary: {(result_df['is_stationary'] == 0).sum()}")
    print(f"  Average length: {result_df['length'].mean():.0f}")
    
    return result_df


def adf_test_predict(series, alpha=0.05):
    """
    Perform ADF test and return prediction.
    
    H0: Unit root exists (non-stationary)
    If p-value < alpha: Reject H0 → Stationary
    If p-value >= alpha: Accept H0 → Non-stationary
    
    Returns:
        0: Stationary
        1: Non-stationary
    """
    try:
        result = adfuller(series, autolag='AIC')
        p_value = result[1]
        
        # p < alpha → stationary (reject H0)
        if p_value < alpha:
            return 0  # Stationary
        else:
            return 1  # Non-stationary
    except Exception as e:
        # If test fails, predict majority class (non-stationary)
        return 1


def kpss_test_predict(series, alpha=0.05, regression='c'):
    """
    Perform KPSS test and return prediction.
    
    H0: Series is stationary
    If p-value < alpha: Reject H0 → Non-stationary
    If p-value >= alpha: Accept H0 → Stationary
    
    Args:
        regression: 'c' (constant) or 'ct' (constant + trend)
    
    Returns:
        0: Stationary
        1: Non-stationary
    """
    try:
        result = kpss(series, regression=regression, nlags='auto')
        p_value = result[1]
        
        # p < alpha → non-stationary (reject H0)
        if p_value < alpha:
            return 1  # Non-stationary
        else:
            return 0  # Stationary
    except Exception as e:
        # If test fails, predict majority class
        return 1


def pp_test_predict(series, alpha=0.05):
    """
    Perform Phillips-Perron test and return prediction.
    
    Similar to ADF but uses different detrending method.
    H0: Unit root exists (non-stationary)
    
    Returns:
        0: Stationary
        1: Non-stationary
    """
    if not PP_AVAILABLE:
        return None
    
    try:
        pp = PhillipsPerron(series)
        p_value = pp.pvalue
        
        # p < alpha → stationary (reject H0)
        if p_value < alpha:
            return 0  # Stationary
        else:
            return 1  # Non-stationary
    except Exception as e:
        return 1


def process_single_series(row, test_func):
    """Process a single time series with given test function."""
    series_id = row.get('series_id', row.name)
    true_label = row['is_stationary']
    
    # Get time series data (already numpy array in 'data' column)
    ts_data = row['data']
    
    # Convert to numpy array if needed
    if not isinstance(ts_data, np.ndarray):
        ts_data = np.array(ts_data)
    
    # Check for valid data
    if len(ts_data) < 10:  # Need minimum length for tests
        return None
    
    # Perform prediction
    try:
        pred = test_func(ts_data)
    except Exception as e:
        # If test fails, return None
        return None
    
    if pred is None:
        return None
    
    # Labels: is_stationary=1 means stationary (0), is_stationary=0 means non-stationary (1)
    true_binary = 0 if true_label == 1 else 1
    
    return {
        'series_id': series_id,
        'true_label': true_binary,
        'predicted': pred,
        'correct': (true_binary == pred)
    }


def compute_baseline_accuracy(df, test_func, test_name, n_jobs=10):
    """Compute accuracy for a given test function."""
    print(f"\n{'='*80}")
    print(f"Running {test_name}...")
    print(f"{'='*80}")
    
    # Split data into chunks for parallel processing
    print(f"Processing {len(df)} series with {n_jobs} workers...")
    chunk_size = max(1, len(df) // n_jobs)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    def process_chunk(chunk_df):
        """Process a chunk of series."""
        results = []
        for _, row in chunk_df.iterrows():
            result = process_single_series(row, test_func)
            if result is not None:
                results.append(result)
        return results
    
    # Use joblib for parallel processing
    chunk_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_chunk)(chunk)
        for chunk in tqdm(chunks, desc=f"{test_name}", unit="chunk")
    )
    
    # Flatten results
    results = [item for sublist in chunk_results for item in sublist]
    
    if not results:
        print(f"No valid results for {test_name}")
        return None
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute metrics
    y_true = results_df['true_label']
    y_pred = results_df['predicted']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'test_name': test_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'n_samples': len(results_df),
        'n_correct': results_df['correct'].sum(),
        'confusion_matrix': cm,
        'results_df': results_df
    }
    
    print(f"\n{test_name} Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Samples:   {len(results_df)}")
    
    return metrics


def plot_comparison(all_metrics, output_dir):
    """Create comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart - Accuracy comparison
    test_names = [m['test_name'] for m in all_metrics if m is not None]
    accuracies = [m['accuracy'] * 100 for m in all_metrics if m is not None]
    
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#27ae60']
    bars = ax1.barh(test_names, accuracies, color=colors[:len(test_names)])
    
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Stationarity Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 100])
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', va='center', fontweight='bold')
    
    # Table - Detailed metrics
    table_data = []
    for m in all_metrics:
        if m is not None:
            table_data.append([
                m['test_name'],
                f"{m['accuracy']*100:.2f}%",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{m['f1_score']:.3f}"
            ])
    
    ax2.axis('tight')
    ax2.axis('off')
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Method', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i == len(table_data):  # Our model (last row)
                table[(i, j)].set_facecolor('#d5f4e6')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#f2f2f2')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'baseline_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison figure saved: {output_file}")
    plt.close()
    
    # Plot confusion matrices (only for tests with CM data)
    tests_with_cm = [m for m in all_metrics if m is not None and m.get('confusion_matrix') is not None]
    
    if tests_with_cm:
        n_tests = len(tests_with_cm)
        fig, axes = plt.subplots(1, n_tests, figsize=(6*n_tests, 5))
        
        if n_tests == 1:
            axes = [axes]
        
        for idx, m in enumerate(tests_with_cm):
            cm = m['confusion_matrix']
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stationary', 'Non-Stat'],
                yticklabels=['Stationary', 'Non-Stat'],
                ax=axes[idx], cbar=True
            )
            
            axes[idx].set_title(f"{m['test_name']}\nAccuracy: {m['accuracy']*100:.2f}%",
                               fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('True')
        
        plt.tight_layout()
        
        # Save confusion matrices
        cm_file = output_dir / 'confusion_matrices.png'
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        print(f"Confusion matrices saved: {cm_file}")
        plt.close()
    else:
        print(f"No confusion matrices to plot")


def save_results(all_metrics, output_dir):
    """Save detailed results to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary table
    summary_data = []
    for m in all_metrics:
        if m is not None:
            summary_data.append({
                'Method': m['test_name'],
                'Accuracy': f"{m['accuracy']*100:.2f}%",
                'Precision': f"{m['precision']:.4f}",
                'Recall': f"{m['recall']:.4f}",
                'F1-Score': f"{m['f1_score']:.4f}",
                'N_Samples': m['n_samples'],
                'N_Correct': m['n_correct']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / 'baseline_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved: {summary_file}")
    
    # Detailed results for each test
    for m in all_metrics:
        if m is not None and 'results_df' in m:
            test_name_safe = m['test_name'].replace(' ', '_').replace('(', '').replace(')', '')
            detail_file = output_dir / f'detailed_{test_name_safe.lower()}.csv'
            m['results_df'].to_csv(detail_file, index=False)
            print(f"Detailed results saved: {detail_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute baseline accuracy for traditional stationarity tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to raw data directory (contains test.parquet)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results and figures')
    parser.add_argument('--n-jobs', type=int, default=10,
                       help='Number of parallel jobs')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level for tests (default: 0.05)')
    parser.add_argument('--sample-size', type=int, default=1000,
                       help='Number of series to sample per pattern (default: 1000, 0=all)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON: Traditional Stationarity Tests")
    print(f"{'='*80}")
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Alpha: {args.alpha}")
    print(f"Sample size: {args.sample_size if args.sample_size > 0 else 'All'}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"{'='*80}\n")
    
    # Load data
    try:
        df = load_test_data(args.data_path, sample_size=args.sample_size)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Run tests
    all_metrics = []
    
    # ADF Test
    adf_metrics = compute_baseline_accuracy(
        df, 
        lambda s: adf_test_predict(s, alpha=args.alpha),
        "ADF Test (p<0.05)",
        n_jobs=args.n_jobs
    )
    all_metrics.append(adf_metrics)
    
    # KPSS Test (constant)
    kpss_metrics = compute_baseline_accuracy(
        df,
        lambda s: kpss_test_predict(s, alpha=args.alpha, regression='c'),
        "KPSS Test (constant)",
        n_jobs=args.n_jobs
    )
    all_metrics.append(kpss_metrics)
    
    # Phillips-Perron Test
    if PP_AVAILABLE:
        pp_metrics = compute_baseline_accuracy(
            df,
            lambda s: pp_test_predict(s, alpha=args.alpha),
            "Phillips-Perron Test",
            n_jobs=args.n_jobs
        )
        all_metrics.append(pp_metrics)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print(f"{'='*80}")
    plot_comparison(all_metrics, args.output_dir)
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    save_results(all_metrics, args.output_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print("BASELINE COMPARISON COMPLETE!")
    print(f"{'='*80}\n")
    
    print("Summary:")
    for m in all_metrics:
        if m is not None:
            print(f"  {m['test_name']:<30} {m['accuracy']*100:>6.2f}%")
    
    if len(all_metrics) > 1:
        baseline_acc = [m['accuracy'] for m in all_metrics[:-1] if m is not None]
        our_acc = all_metrics[-1]['accuracy']
        avg_baseline = np.mean(baseline_acc)
        improvement = (our_acc - avg_baseline) * 100
        
        print(f"\n  Average Baseline:              {avg_baseline*100:>6.2f}%")
        print(f"  Improvement:                   {improvement:>6.2f}% (absolute)")
    
    print(f"\nResults saved in: {args.output_dir}/")
    print(f"  baseline_summary.csv")
    print(f"  baseline_comparison.png")
    print(f"  confusion_matrices.png")
    print(f"  detailed_*.csv\n")


if __name__ == '__main__':
    main()

