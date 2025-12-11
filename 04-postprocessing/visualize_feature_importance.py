#!/usr/bin/env python3
"""
Visualize Feature Importance with Category Analysis

This script loads feature importance from trained models and creates
comprehensive visualizations with feature categorization.

Usage:
    python visualize_feature_importance.py --model model1
    python visualize_feature_importance.py --model model2 --top 30
    python visualize_feature_importance.py --model model1 --classifier XGBoost
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Model configurations
MODEL_CONFIGS = {
    'model1': {
        'name': 'Model 1 (Binary Classification)',
        'saved_models_dir': Path('03-models/hierarchical/model1_binary/saved_models/model1_binary_features'),
    },
    'model2': {
        'name': 'Model 2 (5-Class Classification)',
        'saved_models_dir': Path('03-models/hierarchical/model2_nonstationary/saved_models/model2_nonstationary_features'),
    }
}

# Feature category mapping based on TSFresh feature names
FEATURE_CATEGORIES = {
    'autocorrelation': ['agg_autocorrelation', 'autocorrelation', 'partial_autocorrelation'],
    'statistical': ['variance', 'standard_deviation', 'mean', 'median', 'skewness', 'kurtosis', 
                   'abs_energy', 'absolute_sum_of_changes', 'mean_abs_change', 'mean_change'],
    'stationarity': ['augmented_dickey_fuller', 'c3', 'cid_ce'],
    'frequency': ['fft_coefficient', 'fft_aggregated', 'spkt_welch_density', 'cwt_coefficients'],
    'complexity': ['approximate_entropy', 'sample_entropy', 'binned_entropy', 'svd_entropy',
                  'permutation_entropy', 'lempel_ziv_complexity'],
    'quantile': ['quantile', 'change_quantiles', 'number_crossing_m', 'number_cwt_peaks'],
    'linear': ['agg_linear_trend', 'linear_trend', 'ar_coefficient', 'friedrich_coefficients'],
    'count': ['number_peaks', 'count_above_mean', 'count_below_mean', 'longest_strike',
             'has_duplicate', 'sum_values'],
    'ratio': ['ratio_beyond_r_sigma', 'ratio_value_number_to_time_series_length',
             'percentage_of_reoccurring_datapoints_to_all_datapoints'],
    'range': ['range_count', 'value_count', 'first_location_of_maximum', 
             'last_location_of_maximum', 'first_location_of_minimum', 'last_location_of_minimum'],
    'symmetry': ['symmetry_looking'],
}

# Color mapping for categories (consistent colors across all plots)
CATEGORY_COLORS = {
    'autocorrelation': '#e74c3c',  # Red
    'statistical': '#3498db',      # Blue
    'stationarity': '#2ecc71',     # Green
    'frequency': '#f39c12',        # Orange
    'complexity': '#9b59b6',       # Purple
    'quantile': '#1abc9c',         # Turquoise
    'linear': '#e67e22',           # Dark Orange
    'count': '#34495e',            # Dark Gray
    'ratio': '#16a085',            # Dark Turquoise
    'range': '#c0392b',            # Dark Red
    'symmetry': '#8e44ad',         # Dark Purple
    'other': '#95a5a6',            # Light Gray
}


def categorize_feature(feature_name):
    """Categorize a feature based on its name."""
    feature_lower = feature_name.lower()
    
    for category, patterns in FEATURE_CATEGORIES.items():
        for pattern in patterns:
            if pattern in feature_lower:
                return category
    
    return 'other'


def normalize_importance(df, method='sum'):
    """
    Normalize feature importance to make different models comparable.
    
    Args:
        df: DataFrame with 'importance' column
        method: 'sum' (sum to 1.0) or 'minmax' (scale to 0-1)
    
    Returns:
        DataFrame with additional 'importance_normalized' column
    """
    df = df.copy()
    
    if method == 'sum':
        # Normalize so all importances sum to 1.0 (percentage)
        total = df['importance'].sum()
        if total > 0:
            df['importance_normalized'] = df['importance'] / total
        else:
            df['importance_normalized'] = 0.0
    
    elif method == 'minmax':
        # Scale to 0-1 range
        min_val = df['importance'].min()
        max_val = df['importance'].max()
        if max_val > min_val:
            df['importance_normalized'] = (df['importance'] - min_val) / (max_val - min_val)
        else:
            df['importance_normalized'] = 0.0
    
    return df


def load_feature_importance(model_dir, normalize=True, normalize_method='sum'):
    """
    Load all feature importance CSV files from model directory.
    
    Args:
        model_dir: Path to saved models directory
        normalize: Whether to normalize importance values
        normalize_method: 'sum' (sum to 1.0) or 'minmax' (scale to 0-1)
    """
    model_path = Path(model_dir)
    
    all_importance = {}
    importance_files = list(model_path.glob('feature_importance_*.csv'))
    
    if not importance_files:
        return None
    
    for file in importance_files:
        try:
            df = pd.read_csv(file)
            classifier_name = file.stem.replace('feature_importance_', '')
            
            # Add category column
            df['category'] = df['feature'].apply(categorize_feature)
            
            # Normalize importance if requested
            if normalize:
                df = normalize_importance(df, method=normalize_method)
                # Use normalized values for plotting
                df['importance_raw'] = df['importance'].copy()
                df['importance'] = df['importance_normalized']
            
            all_importance[classifier_name] = df
            print(f"  {classifier_name}: {len(df)} features")
        except Exception as e:
            print(f"  Error loading {file.name}: {e}")
    
    return all_importance


def plot_top_features(all_importance, top_n=20, save_fig=False, output_dir=None, model_name='', show_plot=True, normalized=False):
    """Plot top N features for each classifier."""
    
    n_classifiers = len(all_importance)
    fig, axes = plt.subplots(1, n_classifiers, figsize=(7*n_classifiers, 10))
    
    if n_classifiers == 1:
        axes = [axes]
    
    title_suffix = ' (Normalized)' if normalized else ''
    fig.suptitle(f'Top {top_n} Most Important Features by Classifier{title_suffix}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (classifier_name, df) in enumerate(all_importance.items()):
        ax = axes[idx]
        
        # Get top N features
        top_features = df.nlargest(top_n, 'importance')
        
        # Use global color mapping
        feature_colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['other']) for cat in top_features['category']]
        
        # Horizontal bar plot
        y_pos = np.arange(len(top_features))
        bars = ax.barh(y_pos, top_features['importance'], color=feature_colors)
        
        # Shorten feature names for display
        shortened_names = []
        for feat in top_features['feature']:
            # Remove 'data__' prefix and truncate long names
            short = feat.replace('data__', '').replace('"', '')
            if len(short) > 50:
                short = short[:47] + '...'
            shortened_names.append(short)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(shortened_names, fontsize=9)
        
        xlabel = 'Normalized Importance' if normalized else 'Importance'
        ax.set_xlabel(xlabel, fontsize=12)
        
        total_imp = top_features["importance"].sum()
        if normalized:
            ax.set_title(f'{classifier_name}\nTotal Importance: {total_imp:.3f} ({total_imp*100:.1f}%)',
                        fontsize=13, fontweight='bold')
        else:
            ax.set_title(f'{classifier_name}\nTotal Importance: {total_imp:.3f}',
                        fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Add importance values on bars
        for i, (bar, imp) in enumerate(zip(bars, top_features['importance'])):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f' {imp:.4f}', ha='left', va='center', fontsize=8)
    
    # Create legend for categories (use only categories that appear in the data)
    all_categories = set()
    for df in all_importance.values():
        all_categories.update(df['category'].unique())
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=CATEGORY_COLORS.get(cat, CATEGORY_COLORS['other']), 
                                    label=cat.capitalize()) 
                      for cat in sorted(all_categories)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              bbox_to_anchor=(0.5, -0.02), fontsize=10, title='Feature Categories')
    
    plt.tight_layout()
    
    if save_fig and output_dir:
        model_suffix = f'_{model_name}' if model_name else ''
        output_file = output_dir / f'feature_importance_top{top_n}_comparison{model_suffix}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_category_distribution(all_importance, save_fig=False, output_dir=None, model_name='', show_plot=True, normalized=False):
    """Plot importance distribution by category for each classifier."""
    
    n_classifiers = len(all_importance)
    fig, axes = plt.subplots(2, n_classifiers, figsize=(7*n_classifiers, 12))
    
    if n_classifiers == 1:
        axes = axes.reshape(-1, 1)
    
    title_suffix = ' (Normalized)' if normalized else ''
    fig.suptitle(f'Feature Importance by Category{title_suffix}', fontsize=16, fontweight='bold')
    
    for idx, (classifier_name, df) in enumerate(all_importance.items()):
        # Top subplot: Pie chart
        ax1 = axes[0, idx]
        
        category_importance = df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        # Use global color mapping
        colors = [CATEGORY_COLORS.get(cat, CATEGORY_COLORS['other']) for cat in category_importance.index]
        
        # Calculate percentages
        total = category_importance.sum()
        percentages = (category_importance.values / total) * 100
        
        # Custom autopct function - only show for larger slices
        def autopct_func(pct):
            return f'{pct:.1f}%' if pct >= 8 else ''
        
        # Create pie chart without labels initially
        wedges, texts, autotexts = ax1.pie(category_importance.values, 
                                           labels=None,  # No automatic labels
                                           autopct=autopct_func,
                                           colors=colors,
                                           startangle=90,
                                           pctdistance=0.75,
                                           explode=[0.05 if p < 8 else 0 for p in percentages])  # Explode small slices
        
        # Style percentage texts inside pie
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        # Add legend instead of labels to avoid overlap
        legend_labels = [f'{cat.capitalize()}: {pct:.1f}%' 
                        for cat, pct in zip(category_importance.index, percentages)]
        ax1.legend(wedges, legend_labels, 
                  title="Categories",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1),
                  fontsize=9)
        
        ax1.set_title(f'{classifier_name}\nCategory Distribution', 
                     fontsize=12, fontweight='bold')
        
        # Bottom subplot: Bar chart
        ax2 = axes[1, idx]
        
        y_pos = np.arange(len(category_importance))
        bars = ax2.barh(y_pos, category_importance.values, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([cat.capitalize() for cat in category_importance.index])
        ax2.set_xlabel('Total Importance', fontsize=11)
        ax2.set_title(f'Category Importance Ranking', fontsize=11, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add values on bars
        for bar, val in zip(bars, category_importance.values):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_fig and output_dir:
        model_suffix = f'_{model_name}' if model_name else ''
        output_file = output_dir / f'feature_importance_categories{model_suffix}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_category_summary(all_importance):
    """Print summary statistics by category."""
    
    print(f"\n{'='*80}")
    print("FEATURE IMPORTANCE SUMMARY BY CATEGORY")
    print(f"{'='*80}\n")
    
    for classifier_name, df in all_importance.items():
        print(f"{classifier_name}:")
        print("-" * 80)
        
        category_stats = df.groupby('category')['importance'].agg([
            ('count', 'count'),
            ('total', 'sum'),
            ('mean', 'mean'),
            ('max', 'max')
        ]).sort_values('total', ascending=False)
        
        print(f"{'Category':<20} {'Count':>8} {'Total Imp':>12} {'Mean Imp':>12} {'Max Imp':>12}")
        print("-" * 80)
        
        for category, row in category_stats.iterrows():
            print(f"{category.capitalize():<20} {int(row['count']):>8} "
                  f"{row['total']:>12.4f} {row['mean']:>12.4f} {row['max']:>12.4f}")
        
        print(f"\nTotal features: {len(df)}")
        print(f"Total importance: {df['importance'].sum():.4f}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize feature importance with category analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, choices=['model1', 'model2'], 
                       default='model1',
                       help='Which model to analyze (model1: binary, model2: 5-class)')
    parser.add_argument('--classifier', type=str,
                       help='Specific classifier to analyze (e.g., XGBoost, RandomForest)')
    parser.add_argument('--top', type=int, default=20,
                       help='Number of top features to show (default: 20)')
    parser.add_argument('--saved-models-dir', type=str,
                       help='Path to saved models directory (overrides default)')
    parser.add_argument('--save-fig', action='store_true',
                       help='Save figures to figures/')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, only show statistics')
    parser.add_argument('--normalize', type=str, choices=['sum', 'minmax', 'none'],
                       default='sum',
                       help='Normalize importance: sum (sum to 1.0), minmax (scale to 0-1), none (raw values)')
    
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
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {config['name']}")
    print(f"Saved Models: {saved_models_dir}")
    print(f"Normalization: {args.normalize}")
    print(f"{'='*80}\n")
    
    # Load feature importance
    print(f"[1/3] Loading feature importance files...")
    normalize = (args.normalize != 'none')
    all_importance = load_feature_importance(
        saved_models_dir, 
        normalize=normalize, 
        normalize_method=args.normalize if normalize else 'sum'
    )
    
    if not all_importance:
        print(f"\nError: No feature importance files found in {saved_models_dir}")
        print(f"    Expected files: feature_importance_*.csv")
        sys.exit(1)
    
    # Filter by classifier if specified
    if args.classifier:
        if args.classifier in all_importance:
            all_importance = {args.classifier: all_importance[args.classifier]}
            print(f"\nFiltered to classifier: {args.classifier}")
        else:
            available = ', '.join(all_importance.keys())
            print(f"\nError: Classifier '{args.classifier}' not found")
            print(f"    Available: {available}")
            sys.exit(1)
    
    # Print category summary
    print(f"\n[2/3] Analyzing feature categories...")
    print_category_summary(all_importance)
    
    # Create visualizations
    if args.save_fig or not args.no_plot:
        print(f"\n[3/3] Generating visualizations...")
        
        output_dir = Path('figures') if args.save_fig else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        show_plot = not args.no_plot
        is_normalized = (args.normalize != 'none')
        
        # Plot top features
        plot_top_features(all_importance, args.top, args.save_fig, output_dir, args.model, show_plot, is_normalized)
        
        # Plot category distribution
        plot_category_distribution(all_importance, args.save_fig, output_dir, args.model, show_plot, is_normalized)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
