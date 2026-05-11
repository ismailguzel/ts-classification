#!/usr/bin/env python3
"""
Visualize Misclassified Time Series

This script loads a time series by ID and visualizes it along with:
- True label
- Predicted label
- Prediction confidence
- Time series plot with statistical properties

Usage:
    python visualize_misclassified.py --id 15255 --model model1
    python visualize_misclassified.py --id 17071 --model model1 --save-fig
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

_NAMES = [
    'stationary', 'deterministic_trend', 'stochastic_trend', 'volatility',
    'collective_anomaly', 'contextual_anomaly', 'mean_shift', 'point_anomaly',
    'trend_shift', 'variance_shift', 'cubic_collective', 'cubic_mean_shift',
    'cubic_point_anomaly', 'cubic_variance_shift', 'damped_collective',
    'damped_mean_shift', 'damped_point_anomaly', 'damped_variance_shift',
    'exponential_collective', 'exponential_mean_shift', 'exponential_point_anomaly',
    'exponential_variance_shift', 'linear_collective', 'linear_mean_shift',
    'linear_point_anomaly', 'linear_trend_shift', 'linear_variance_shift',
    'quadratic_collective', 'quadratic_mean_shift', 'quadratic_point_anomaly',
    'quadratic_variance_shift', 'stochastic_collective', 'stochastic_mean_shift',
    'stochastic_point_anomaly', 'stochastic_variance_shift', 'volatility_collective',
    'volatility_mean_shift', 'volatility_point_anomaly', 'volatility_variance_shift',
]

# Model configurations
MODEL_CONFIGS = {
    'model1': {
        'name': 'Flat Classifier',
        'saved_models_dir': Path('03-models/flat_classifier/output'),
        'raw_data_dir': Path('data/raw/dataset'),
        'class_names': {i: n for i, n in enumerate(_NAMES)},
    },
}


def load_time_series(series_id, raw_data_dir):
    """Load time series data from parquet files."""
    raw_data_path = Path(raw_data_dir)
    
    # Search for the series ID in all parquet files
    parquet_files = list(raw_data_path.rglob('*.parquet'))
    
    for parquet_file in parquet_files:
        try:
            # Read with filter to reduce memory usage
            df = pd.read_parquet(
                parquet_file,
                filters=[('series_id', '==', series_id)]
            )
            
            if len(df) > 0:
                # Data is in long format - group by series_id and collect time series
                series_data = df.sort_values('time')
                ts_data = series_data['data'].values
                
                # Get metadata from first row
                first_row = series_data.iloc[0]
                if 'primary_category' in first_row:
                    category = f"{first_row['primary_category']}/{first_row.get('sub_category', 'unknown')}"
                else:
                    category = parquet_file.parent.name
                
                return ts_data, category
                
        except Exception as e:
            # If filtering doesn't work, try reading entire file
            try:
                df = pd.read_parquet(parquet_file)
                id_col = 'series_id' if 'series_id' in df.columns else 'id'
                
                if id_col in df.columns and series_id in df[id_col].values:
                    series_data = df[df[id_col] == series_id].sort_values('time')
                    ts_data = series_data['data'].values
                    
                    first_row = series_data.iloc[0]
                    if 'primary_category' in first_row:
                        category = f"{first_row['primary_category']}/{first_row.get('sub_category', 'unknown')}"
                    else:
                        category = parquet_file.parent.name
                    
                    return ts_data, category
            except Exception as e2:
                continue
    
    return None, None


def load_prediction_info(series_id, model_dir, model_name):
    """Load prediction information from misclassified CSV or predictions CSV."""
    model_path = Path(model_dir)
    
    # Try misclassified files first
    misclassified_files = list(model_path.glob('misclassified_*.csv'))
    for file in misclassified_files:
        try:
            df = pd.read_csv(file)
            if series_id in df['id'].values:
                row = df[df['id'] == series_id].iloc[0]
                classifier_name = file.stem.replace('misclassified_', '')
                return row.to_dict(), classifier_name, True
        except Exception as e:
            continue
    
    # Try predictions files
    prediction_files = list(model_path.glob('predictions_*.csv'))
    for file in prediction_files:
        try:
            df = pd.read_csv(file)
            if series_id in df['id'].values:
                row = df[df['id'] == series_id].iloc[0]
                classifier_name = file.stem.replace('predictions_', '')
                is_misclassified = row['correct'] == 0
                return row.to_dict(), classifier_name, is_misclassified
        except Exception as e:
            continue
    
    return None, None, None


def compute_statistics(ts_data):
    """Compute statistical properties of time series."""
    return {
        'mean': np.mean(ts_data),
        'std': np.std(ts_data),
        'min': np.min(ts_data),
        'max': np.max(ts_data),
        'median': np.median(ts_data),
        'length': len(ts_data),
        'trend': np.polyfit(range(len(ts_data)), ts_data, 1)[0]  # Linear trend slope
    }


def plot_time_series(ts_data, series_id, pred_info, classifier_name, 
                     category, stats, class_names, model_name, save_fig=False):
    """Create visualization of the time series with prediction information."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Main time series plot
    ax1 = axes[0]
    time_index = np.arange(len(ts_data))
    
    # Plot the series
    ax1.plot(time_index, ts_data, linewidth=1.5, color='#2E86AB', alpha=0.8)
    ax1.fill_between(time_index, ts_data, alpha=0.2, color='#2E86AB')
    
    # Add trend line if significant
    if abs(stats['trend']) > 1e-6:
        trend_line = stats['trend'] * time_index + (stats['mean'] - stats['trend'] * len(ts_data) / 2)
        ax1.plot(time_index, trend_line, '--', color='red', linewidth=2, 
                label=f'Trend (slope: {stats["trend"]:.6f})', alpha=0.7)
    
    # Add mean line
    ax1.axhline(y=stats['mean'], color='green', linestyle='--', 
               linewidth=1.5, alpha=0.6, label=f'Mean: {stats["mean"]:.2f}')
    
    # Title and labels
    true_label = class_names.get(pred_info['true_label'], pred_info['true_label'])
    pred_label = class_names.get(pred_info['predicted_label'], pred_info['predicted_label'])
    
    title = f"Time Series ID: {series_id} | Category: {category}\n"
    title += f"True Label: {true_label} | Predicted: {pred_label} | "
    title += f"Confidence: {pred_info['confidence']:.4f}"
    
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time Index', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Statistics panel
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create statistics text
    stats_text = f"""
    Model: {model_name}
    Classifier: {classifier_name}
    
    Statistical Properties:
    • Length: {stats['length']}
    • Mean: {stats['mean']:.4f}
    • Std Dev: {stats['std']:.4f}
    • Median: {stats['median']:.4f}
    • Min: {stats['min']:.4f}
    • Max: {stats['max']:.4f}
    • Trend Slope: {stats['trend']:.6f}
    """
    
    # Prediction probabilities
    prob_text = "\n    Prediction Probabilities:\n"
    for key, value in pred_info.items():
        if key.startswith('prob_'):
            class_name = key.replace('prob_', '')
            prob_text += f"    • {class_name}: {value:.4f}\n"
    
    stats_text += prob_text
    
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')
    
    plt.tight_layout()
    
    if save_fig:
        output_dir = Path('figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'misclassified_{series_id}_{classifier_name}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize misclassified time series with predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a misclassified sample from Model 1
  python visualize_misclassified.py --id 15255 --model model1
  
  # Visualize from Model 2 and save figure
  python visualize_misclassified.py --id 17071 --model model1 --save-fig
  
  # Use custom paths
  python visualize_misclassified.py --id 3344 --model model1 \\
      --saved-models-dir custom/path \\
      --raw-data-dir custom/raw/data
        """
    )
    
    parser.add_argument('--id', type=int, required=True,
                       help='Time series ID to visualize')
    parser.add_argument('--model', type=str, choices=['model1'],
                       default='model1',
                       help='Which model to use')
    parser.add_argument('--saved-models-dir', type=str,
                       help='Path to saved models directory (overrides default)')
    parser.add_argument('--raw-data-dir', type=str,
                       help='Path to raw data directory (overrides default)')
    parser.add_argument('--save-fig', action='store_true',
                       help='Save figure to figures/')
    
    args = parser.parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    
    # Override paths if provided
    if args.saved_models_dir:
        config['saved_models_dir'] = Path(args.saved_models_dir)
    if args.raw_data_dir:
        config['raw_data_dir'] = Path(args.raw_data_dir)
    
    # Make paths absolute
    project_root = Path(__file__).parent.parent
    saved_models_dir = project_root / config['saved_models_dir']
    raw_data_dir = project_root / config['raw_data_dir']
    
    print(f"\n{'='*80}")
    print(f"VISUALIZING TIME SERIES: {args.id}")
    print(f"{'='*80}")
    print(f"Model: {config['name']}")
    print(f"Saved Models: {saved_models_dir}")
    print(f"Raw Data: {raw_data_dir}")
    print(f"{'='*80}\n")
    
    # Load prediction information
    print(f"[1/3] Loading prediction information...")
    pred_info, classifier_name, is_misclassified = load_prediction_info(
        args.id, saved_models_dir, args.model
    )
    
    if pred_info is None:
        print(f"Error: Series ID {args.id} not found in predictions/misclassified files")
        print(f"  Searched in: {saved_models_dir}")
        sys.exit(1)
    
    status = "MISCLASSIFIED" if is_misclassified else "CORRECTLY CLASSIFIED"
    print(f"Found prediction: {status} by {classifier_name}")
    
    # Load time series data
    print(f"\n[2/3] Loading time series data...")
    ts_data, category = load_time_series(args.id, raw_data_dir)
    
    if ts_data is None:
        print(f"Error: Series ID {args.id} not found in raw data")
        print(f"  Searched in: {raw_data_dir}")
        sys.exit(1)
    
    print(f"Loaded time series from category: {category}")
    print(f"  Length: {len(ts_data)}")
    
    # Compute statistics
    print(f"\n[3/3] Computing statistics and generating plot...")
    stats = compute_statistics(ts_data)
    
    # Plot
    plot_time_series(
        ts_data, args.id, pred_info, classifier_name,
        category, stats, config['class_names'], 
        config['name'], args.save_fig
    )
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
