#!/usr/bin/env python3
"""
Extract Feature Importance from Already Trained AutoGluon Models
"""
import sys
from pathlib import Path
import pandas as pd

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / '03-models'))

# Change to project root
import os
os.chdir(Path(__file__).parent)

print("\n" + "="*80)
print("EXTRACTING FEATURE IMPORTANCE FROM TRAINED AUTOGLUON MODELS")
print("="*80 + "\n")

# Model configurations
models = [
    {
        'name': 'Model 1 (Binary)',
        'autogluon_path': Path('03-models/hierarchical/model1_binary/saved_models/model1_binary_autogluon'),
        'features_path': Path('data/features/unified-20k/selected'),
        'target': 'binary'
    },
    {
        'name': 'Model 2 (5-Class)',
        'autogluon_path': Path('03-models/hierarchical/model2_nonstationary/saved_models/model2_nonstationary_autogluon'),
        'features_path': Path('data/features/unified-20k/selected'),
        'target': 'primary'
    }
]

try:
    from autogluon.tabular import TabularPredictor
    from utils import load_features_and_labels, split_train_test
    
    for model_config in models:
        print(f"\n{'='*80}")
        print(f"Processing: {model_config['name']}")
        print(f"{'='*80}\n")
        
        autogluon_path = model_config['autogluon_path']
        
        # Check if model exists
        if not autogluon_path.exists():
            print(f"⚠ Model not found: {autogluon_path}")
            print(f"  Skipping...\n")
            continue
        
        # Check if feature importance already exists
        fi_path = autogluon_path / "feature_importance_AutoGluon.csv"
        if fi_path.exists():
            print(f"✓ Feature importance already exists: {fi_path}")
            print(f"  Skipping...\n")
            continue
        
        print(f"[1/3] Loading AutoGluon predictor...")
        try:
            predictor = TabularPredictor.load(str(autogluon_path))
            print(f"✓ Predictor loaded from: {autogluon_path}")
        except Exception as e:
            print(f"✗ Failed to load predictor: {e}")
            continue
        
        print(f"\n[2/3] Loading test data...")
        try:
            # Load features
            X_df, y, _ = load_features_and_labels(
                model_config['features_path'], 
                target=model_config['target'], 
                verbose=False
            )
            
            # Split to get test set (same split as training)
            _, X_test_df, _, y_test, _, _ = split_train_test(
                X_df, y, 
                sample_ids=X_df.index.to_numpy(),
                test_size=0.2,
                random_state=42,
                stratify=True
            )
            
            # Create test DataFrame in AutoGluon format
            test_df = X_test_df.copy()
            test_df['_label'] = y_test
            
            print(f"✓ Test data loaded: {len(test_df)} samples, {len(X_test_df.columns)} features")
        except Exception as e:
            print(f"✗ Failed to load test data: {e}")
            continue
        
        print(f"\n[3/3] Extracting feature importance...")
        try:
            # Get feature importance
            feature_importance = predictor.feature_importance(test_df)
            
            if feature_importance is not None and not feature_importance.empty:
                # Sort by importance and get top 50
                if 'importance' in feature_importance.columns:
                    feature_importance_sorted = feature_importance.sort_values(
                        by='importance', ascending=False
                    ).head(50)
                    
                    # Keep only feature name and importance
                    fi_df = feature_importance_sorted[['importance']].copy()
                    fi_df.index.name = 'feature_name'
                    
                    # Save to CSV
                    fi_df.to_csv(fi_path)
                    print(f"✓ Feature importance saved to: {fi_path}")
                    print(f"\n  Top 5 features:")
                    for idx, (feat, row) in enumerate(feature_importance_sorted.head(5).iterrows(), 1):
                        feat_display = feat.replace('data__', '').replace('__', ' ')[:60]
                        print(f"    {idx}. {feat_display}: {row['importance']:.4f}")
                else:
                    print(f"✗ 'importance' column not found in DataFrame")
            else:
                print(f"✗ Feature importance is empty or None")
        except Exception as e:
            print(f"✗ Failed to extract feature importance: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("✓ EXTRACTION COMPLETE")
    print(f"{'='*80}\n")

except ImportError as e:
    print(f"✗ ImportError: {e}")
    print(f"  Make sure you're in the ts-autogluon environment")
    print(f"  Run: conda activate ts-autogluon")
    sys.exit(1)
