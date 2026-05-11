#!/usr/bin/env python3
"""
Generate Confusion Matrices for Trained Models

This script creates high-quality confusion matrices for all trained classifiers
from prediction CSV files.

Usage:
    python visualize_confusion_matrices.py --model model1
    python visualize_confusion_matrices.py --model model1 --save-fig
    python visualize_confusion_matrices.py --model both --format png
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

CLASS_NAMES_39 = [
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
        'output_dir': Path('03-models/flat_classifier/output'),
        'class_names': CLASS_NAMES_39,
        'classifiers': ['RandomForest', 'XGBoost', 'CatBoost', 'SVM_RBF']
    },
}


def load_predictions(output_dir, classifier_name):
    """Load predictions from CSV file."""
    pred_file = output_dir / f'predictions_{classifier_name}.csv'
    
    if not pred_file.exists():
        print(f"  Warning: {pred_file.name} not found")
        return None
    
    try:
        df = pd.read_csv(pred_file)
        return df
    except Exception as e:
        print(f"  Error loading {pred_file.name}: {e}")
        return None


def create_confusion_matrix(df):
    """Generate confusion matrix from predictions DataFrame."""
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix_single(cm, class_names, classifier_name, model_name, 
                                 normalize=False, save_fig=False, output_dir=None, fmt='png'):
    """Plot a single confusion matrix with large, clear annotations."""
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt_str = '.2%'
        title = f'{model_name}\n{classifier_name} - Normalized Confusion Matrix'
        cmap = 'Blues'
    else:
        cm_display = cm
        fmt_str = 'd'
        title = f'{model_name}\n{classifier_name} - Confusion Matrix'
        cmap = 'Blues'
    
    # Calculate figure size based on number of classes
    n_classes = len(class_names)
    figsize = (max(8, n_classes * 1.5), max(6, n_classes * 1.3))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm_display, annot=True, fmt=fmt_str, cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    # Rotate x-axis labels if needed
    if n_classes > 3:
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(class_names, rotation=0, fontsize=12)
    else:
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_yticklabels(class_names, fontsize=12)
    
    plt.tight_layout()
    
    if save_fig and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_tag = model_name.lower().replace(' ', '_').replace(':', '')
        norm_tag = '_normalized' if normalize else ''
        filename = f'cm_{model_tag}_{classifier_name.lower()}{norm_tag}.{fmt}'
        output_file = output_dir / filename
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file.name}")
    
    plt.close()


def plot_confusion_matrices_grid(all_cms, class_names, classifiers, model_name,
                                 normalize=False, save_fig=False, output_dir=None, fmt='png'):
    """Plot all confusion matrices in a grid layout."""
    
    n_classifiers = len(classifiers)
    n_cols = 2
    n_rows = (n_classifiers + 1) // 2
    
    # Calculate figure size
    n_classes = len(class_names)
    fig_width = n_cols * max(6, n_classes * 1.2)
    fig_height = n_rows * max(5, n_classes * 1.1)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Flatten axes for easy iteration
    if n_classifiers == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for idx, (classifier, cm) in enumerate(zip(classifiers, all_cms)):
        if cm is None:
            axes[idx].axis('off')
            continue
        
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt_str = '.2f'
            cmap = 'Blues'
        else:
            cm_display = cm
            fmt_str = 'd'
            cmap = 'Blues'
        
        # Create heatmap
        sns.heatmap(cm_display, annot=True, fmt=fmt_str, cmap=cmap,
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                    annot_kws={'fontsize': 13, 'fontweight': 'bold'},
                    linewidths=0.5, linecolor='gray')
        
        axes[idx].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True', fontsize=12, fontweight='bold')
        axes[idx].set_title(f'{classifier}', fontsize=14, fontweight='bold', pad=10)
        
        # Adjust tick labels
        if n_classes > 3:
            axes[idx].set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
            axes[idx].set_yticklabels(class_names, rotation=0, fontsize=10)
        else:
            axes[idx].tick_params(labelsize=10)
    
    # Hide unused subplots
    for idx in range(n_classifiers, len(axes)):
        axes[idx].axis('off')
    
    norm_tag = ' (Normalized)' if normalize else ''
    fig.suptitle(f'{model_name} - Confusion Matrices{norm_tag}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_fig and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_tag = model_name.lower().replace(' ', '_').replace(':', '')
        norm_tag = '_normalized' if normalize else ''
        filename = f'cm_grid_{model_tag}{norm_tag}.{fmt}'
        output_file = output_dir / filename
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Saved grid: {output_file.name}")
    
    plt.close()


def process_model(model_key, normalize=False, save_fig=False,
                 individual=True, grid=True, fmt='png', figures_dir=None):
    """Process confusion matrices for a specific model."""

    config = MODEL_CONFIGS[model_key]

    print(f"\n{'='*80}")
    print(f"Processing {config['name']}")
    print(f"{'='*80}\n")

    project_root = Path(__file__).parent.parent
    cfg_dir = config['output_dir']
    output_dir = cfg_dir if cfg_dir.is_absolute() else project_root / cfg_dir
    if figures_dir is None:
        figures_dir = project_root / '04-postprocessing' / 'figures'
    else:
        figures_dir = Path(figures_dir)
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    print(f"Loading predictions from: {output_dir}")

    # Determine actual class names from first available prediction file
    all_class_names = config['class_names']
    present_class_names = all_class_names  # will be refined below

    all_cms = []
    valid_classifiers = []

    for classifier in config['classifiers']:
        print(f"\n  {classifier}:")
        df = load_predictions(output_dir, classifier)

        if df is None:
            all_cms.append(None)
            continue

        # Derive class names from labels present in this file
        present_labels = sorted(set(df['true_label']) | set(df['predicted_label']))
        if max(present_labels) < len(all_class_names):
            present_class_names = [all_class_names[i] for i in present_labels]
        else:
            present_class_names = [str(i) for i in present_labels]

        cm = create_confusion_matrix(df)
        all_cms.append(cm)
        valid_classifiers.append(classifier)

        print(f"    Loaded {len(df)} predictions  ({len(present_labels)} classes)")
        print(f"    Accuracy: {(df['true_label'] == df['predicted_label']).mean():.4f}")

        if individual:
            plot_confusion_matrix_single(cm, present_class_names, classifier,
                                        config['name'], normalize=normalize,
                                        save_fig=save_fig, output_dir=figures_dir, fmt=fmt)

    # Plot grid of all confusion matrices
    if grid and valid_classifiers:
        print(f"\nGenerating grid visualization...")
        valid_cms = [cm for cm in all_cms if cm is not None]
        plot_confusion_matrices_grid(valid_cms, present_class_names,
                                    valid_classifiers, config['name'],
                                    normalize=normalize, save_fig=save_fig,
                                    output_dir=figures_dir, fmt=fmt)
    
    print(f"\n{'='*80}")
    print(f"Completed {config['name']}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrices from model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Model 1 with default settings
  python visualize_confusion_matrices.py --model model1
  
  # Model 2 with normalized matrices and save
  python visualize_confusion_matrices.py --model model1 --normalize --save-fig
  
  # Both models, grid only, save as PDF
  python visualize_confusion_matrices.py --model both --grid-only --save-fig --format pdf
        """
    )
    
    parser.add_argument('--model', type=str, choices=['model1'],
                       default='model1',
                       help='Which model to process')
    parser.add_argument('--saved-models-dir', type=str, default=None,
                       help='Path to saved models directory (overrides default)')
    parser.add_argument('--normalize', action='store_true',
                       help='Show normalized confusion matrices (proportions)')
    parser.add_argument('--save-fig', action='store_true',
                       help='Save figures to 04-postprocessing/figures/')
    parser.add_argument('--figures-dir', type=str, default=None,
                       help='Directory to save figures (overrides default figures/)')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'svg'], 
                       default='png',
                       help='Output figure format (default: png)')
    parser.add_argument('--grid-only', action='store_true',
                       help='Only generate grid visualization, skip individual plots')
    parser.add_argument('--individual-only', action='store_true',
                       help='Only generate individual plots, skip grid')
    
    args = parser.parse_args()
    
    # Determine what to generate
    individual = not args.grid_only
    grid = not args.individual_only
    
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX VISUALIZATION")
    print(f"{'='*80}")
    print(f"Model(s): {args.model}")
    print(f"Normalized: {args.normalize}")
    print(f"Save figures: {args.save_fig}")
    print(f"Format: {args.format}")
    print(f"Individual plots: {individual}")
    print(f"Grid plots: {grid}")
    print(f"{'='*80}")
    
    if args.saved_models_dir:
        MODEL_CONFIGS[args.model]['output_dir'] = Path(args.saved_models_dir)

    # Process model
    if True:
        process_model(args.model, normalize=args.normalize, save_fig=args.save_fig,
                     individual=individual, grid=grid, fmt=args.format,
                     figures_dir=args.figures_dir)

    if args.save_fig:
        figures_dir = Path(args.figures_dir) if args.figures_dir else Path(__file__).parent / 'figures'
        print(f"\n{'='*80}")
        print(f"All figures saved to: {figures_dir}")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("Note: Use --save-fig to save figures to disk")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
