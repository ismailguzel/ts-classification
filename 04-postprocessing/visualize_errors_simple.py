#!/usr/bin/env python3
"""
Visualize Misclassified Samples - Simple Version

This script analyzes misclassified samples from prediction CSV files without
requiring raw time series data.

Usage:
    python visualize_errors_simple.py --model model1
    python visualize_errors_simple.py --model model2 --top 20
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Model configurations
MODEL_CONFIGS = {
    'model1': {
        'name': 'Model 1 (Binary Classification)',
        'saved_models_dir': Path('03-models/hierarchical/model1_binary/saved_models/model1_binary_features'),
        'class_names': {0: 'Stationary', 1: 'Non-Stationary'}
    },
    'model2': {
        'name': 'Model 2 (5-Class Classification)',
        'saved_models_dir': Path('03-models/hierarchical/model2_nonstationary/saved_models/model2_nonstationary_features'),
        'class_names': {
            0: 'Trend',
            1: 'Volatility',
            2: 'Stochastic',
            3: 'Anomaly',
            4: 'Structural Break'
        }
    }
}


def load_misclassified_data(model_dir):
    """Load all misclassified samples from all classifiers."""
    model_path = Path(model_dir)
    
    all_misclassified = {}
    misclassified_files = list(model_path.glob('misclassified_*.csv'))
    
    for file in misclassified_files:
        try:
            df = pd.read_csv(file)
            classifier_name = file.stem.replace('misclassified_', '')
            all_misclassified[classifier_name] = df
            print(f"  ✓ {classifier_name}: {len(df)} misclassified samples")
        except Exception as e:
            print(f"  ✗ Error loading {file.name}: {e}")
    
    return all_misclassified


def plot_error_distribution(all_misclassified, class_names, model_name, save_fig=False):
    """Plot error distribution across classifiers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{model_name} - Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. Error counts by classifier
    ax1 = axes[0, 0]
    error_counts = {name: len(df) for name, df in all_misclassified.items()}
    classifiers = list(error_counts.keys())
    counts = list(error_counts.values())
    
    colors = sns.color_palette("husl", len(classifiers))
    ax1.barh(classifiers, counts, color=colors)
    ax1.set_xlabel('Number of Misclassified Samples', fontsize=12)
    ax1.set_title('Misclassification Count by Classifier', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    for i, count in enumerate(counts):
        ax1.text(count, i, f' {count}', va='center', fontsize=10)
    
    # 2. Confusion patterns (True vs Predicted)
    ax2 = axes[0, 1]
    
    # Aggregate all misclassifications
    all_errors = pd.concat([df.assign(classifier=name) 
                           for name, df in all_misclassified.items()])
    
    confusion_data = all_errors.groupby(['true_label', 'predicted_label']).size().unstack(fill_value=0)
    
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Count'})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title('Aggregated Confusion Matrix (Errors Only)', fontsize=13, fontweight='bold')
    
    # 3. Confidence distribution for errors
    ax3 = axes[1, 0]
    
    for name, df in all_misclassified.items():
        ax3.hist(df['confidence'], bins=20, alpha=0.6, label=name, edgecolor='black')
    
    ax3.set_xlabel('Prediction Confidence', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Confidence Distribution for Misclassified Samples', fontsize=13, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)
    
    # 4. Error rate by true label
    ax4 = axes[1, 1]
    
    true_label_errors = all_errors['true_label'].value_counts().sort_index()
    
    labels = [class_names.get(label, f'Class {label}') for label in true_label_errors.index]
    ax4.bar(range(len(labels)), true_label_errors.values, color=colors[:len(labels)])
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_ylabel('Number of Errors', fontsize=12)
    ax4.set_title('Errors by True Label', fontsize=13, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(true_label_errors.values):
        ax4.text(i, count, f'{count}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        output_dir = Path('figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'error_analysis_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Figure saved to: {output_file}")
    
    plt.show()


def print_worst_cases(all_misclassified, class_names, top_n=10):
    """Print worst misclassified cases (highest confidence errors)."""
    
    print(f"\n{'='*80}")
    print(f"TOP {top_n} WORST MISCLASSIFICATIONS (Highest Confidence Errors)")
    print(f"{'='*80}\n")
    
    for classifier_name, df in all_misclassified.items():
        if len(df) == 0:
            continue
            
        print(f"\n{classifier_name}:")
        print("-" * 80)
        
        # Sort by confidence (descending)
        worst = df.nlargest(top_n, 'confidence')
        
        for idx, row in worst.iterrows():
            true_name = class_names.get(row['true_label'], row['true_label'])
            pred_name = class_names.get(row['predicted_label'], row['predicted_label'])
            
            print(f"  ID: {int(row['id']):6d} | True: {true_name:20s} → Predicted: {pred_name:20s} | "
                  f"Confidence: {row['confidence']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze misclassified samples across classifiers',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, choices=['model1', 'model2'], 
                       default='model1',
                       help='Which model to analyze (model1: binary, model2: 5-class)')
    parser.add_argument('--saved-models-dir', type=str,
                       help='Path to saved models directory (overrides default)')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of worst cases to show (default: 10)')
    parser.add_argument('--save-fig', action='store_true',
                       help='Save figures to figures/')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, only show statistics')
    
    args = parser.parse_args()
    
    # Get model configuration
    config = MODEL_CONFIGS[args.model]
    
    # Override paths if provided
    if args.saved_models_dir:
        config['saved_models_dir'] = Path(args.saved_models_dir)
    
    # Make paths absolute
    project_root = Path(__file__).parent.parent
    saved_models_dir = project_root / config['saved_models_dir']
    
    print(f"\n{'='*80}")
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {config['name']}")
    print(f"Saved Models: {saved_models_dir}")
    print(f"{'='*80}\n")
    
    # Load misclassified data
    print(f"[1/3] Loading misclassified samples...")
    all_misclassified = load_misclassified_data(saved_models_dir)
    
    if not all_misclassified:
        print(f"\n✗ Error: No misclassified files found in {saved_models_dir}")
        sys.exit(1)
    
    total_errors = sum(len(df) for df in all_misclassified.values())
    print(f"\n✓ Total misclassified samples: {total_errors}")
    
    # Print worst cases
    print(f"\n[2/3] Analyzing worst cases...")
    print_worst_cases(all_misclassified, config['class_names'], args.top)
    
    # Plot distributions
    if not args.no_plot:
        print(f"\n[3/3] Generating visualizations...")
        plot_error_distribution(all_misclassified, config['class_names'], 
                               config['name'], args.save_fig)
    
    print(f"\n{'='*80}")
    print("✓ Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
