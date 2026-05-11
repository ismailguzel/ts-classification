"""
Model evaluation utilities for time series classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report as sklearn_classification_report,
    confusion_matrix
)
import json
from datetime import datetime
from pathlib import Path


class ModelEvaluator:
    """Comprehensive model evaluation with automatic CSV exports."""
    
    def __init__(self, model_name, model, class_names=None, output_dir=None):
        """Initialize model evaluator."""
        self.model_name = model_name
        self.model = model
        self.class_names = class_names or []
        self.results = {}
        self.train_time = None
        self.feature_names = None
        self.output_dir = Path(output_dir) if output_dir else None
    
    def set_train_time(self, train_time):
        """Set training time in seconds."""
        self.train_time = train_time
        return self
    
    def set_feature_names(self, feature_names):
        """Set feature names for reporting."""
        self.feature_names = feature_names
        return self
    
    def evaluate(self, X_train, y_train, X_test, y_test, train_ids=None, test_ids=None):
        """Compute comprehensive evaluation metrics."""
        print(f"\n{'='*80}")
        print(f"EVALUATING MODEL: {self.model_name}")
        print(f"{'='*80}")
        
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        try:
            y_train_proba = self.model.predict_proba(X_train)
            y_test_proba = self.model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            y_train_proba = None
            y_test_proba = None
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train, y_train_pred, average='weighted', zero_division=0
        )
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted', zero_division=0
        )
        
        present_labels = sorted(set(y_train) | set(y_test))
        present_names = [self.class_names[i] for i in present_labels] if self.class_names else None

        train_report = sklearn_classification_report(
            y_train, y_train_pred, labels=present_labels, target_names=present_names,
            output_dict=True, zero_division=0
        )
        test_report = sklearn_classification_report(
            y_test, y_test_pred, labels=present_labels, target_names=present_names,
            output_dict=True, zero_division=0
        )
        
        train_cm = confusion_matrix(y_train, y_train_pred).tolist()
        test_cm = confusion_matrix(y_test, y_test_pred).tolist()
        
        print(f"\n ACCURACY:")
        print(f"  Train: {train_acc:.4f}")
        print(f"  Test:  {test_acc:.4f}")
        print(f"\n WEIGHTED METRICS:")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall:    {test_recall:.4f}")
        print(f"  Test F1:        {test_f1:.4f}")
        
        if self.output_dir and test_ids is not None and y_test_proba is not None:
            self._save_predictions(test_ids, y_test, y_test_pred, y_test_proba)
        
        # Extract feature importance if available
        feature_importance = None
        if self.feature_names is not None:
            feature_importance = self.extract_feature_importance()
        
        self.results = {
            'model_name': self.model_name,
            'train_time': self.train_time,
            'metrics': {
                'train': {
                    'accuracy': float(train_acc),
                    'precision': float(train_precision),
                    'recall': float(train_recall),
                    'f1': float(train_f1),
                },
                'test': {
                    'accuracy': float(test_acc),
                    'precision': float(test_precision),
                    'recall': float(test_recall),
                    'f1': float(test_f1),
                }
            },
            'classification_reports': {'train': train_report, 'test': test_report},
            'confusion_matrices': {'train': train_cm, 'test': test_cm},
            'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds')
        }
        
        if feature_importance:
            self.results['feature_importance'] = feature_importance
        
        print(f"{'='*80}\n")
        return self.results
    
    def _save_predictions(self, test_ids, y_true, y_pred, y_proba):
        """Save predictions with probabilities to CSV."""
        n_classes = y_proba.shape[1]
        
        # Ensure all arrays are 1-dimensional
        test_ids = np.asarray(test_ids).ravel()
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        
        pred_data = {
            'id': test_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
            'correct': (y_true == y_pred).astype(int)
        }
        
        for i in range(n_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            pred_data[f'prob_{class_name}'] = y_proba[:, i]
        
        pred_data['confidence'] = np.max(y_proba, axis=1)
        pred_df = pd.DataFrame(pred_data)
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_file = self.output_dir / f'predictions_{self.model_name}.csv'
            misclassified_file = self.output_dir / f'misclassified_{self.model_name}.csv'
        else:
            output_file = f'predictions_{self.model_name}.csv'
            misclassified_file = f'misclassified_{self.model_name}.csv'
        
        pred_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")
        
        misclassified_df = pred_df[pred_df['correct'] == 0].copy()
        if not misclassified_df.empty:
            misclassified_df.to_csv(misclassified_file, index=False)
            print(f"Misclassified samples saved to: {misclassified_file} ({len(misclassified_df)} errors)")
        
        self.results['prediction_files'] = {
            'predictions_csv': str(output_file),
            'misclassified_csv': str(misclassified_file) if not misclassified_df.empty else None,
            'misclassified_count': len(misclassified_df)
        }
    
    def extract_feature_importance(self):
        """Extract feature importance from tree-based models."""
        importance_dict = None
        
        # Try different attribute names for feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'get_feature_importance'):
            # CatBoost specific
            try:
                importances = self.model.get_feature_importance()
            except:
                importances = None
        else:
            importances = None
        
        if importances is not None and self.feature_names is not None:
            if len(importances) == len(self.feature_names):
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                
                importance_dict = {
                    'features': [self.feature_names[i] for i in indices],
                    'importances': [float(importances[i]) for i in indices]
                }
                
                # Save top features to separate CSV
                if self.output_dir:
                    top_n = min(50, len(indices))  # Top 50 features
                    importance_df = pd.DataFrame({
                        'feature': [self.feature_names[i] for i in indices[:top_n]],
                        'importance': [importances[i] for i in indices[:top_n]],
                        'rank': range(1, top_n + 1)
                    })
                    
                    importance_file = self.output_dir / f'feature_importance_{self.model_name}.csv'
                    importance_df.to_csv(importance_file, index=False)
                    print(f"Feature importance saved to: {importance_file}")
        
        return importance_dict
    
    def save_results(self, output_path):
        """Save complete results to JSON."""
        output_path = Path(output_path)
        
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / f'{self.model_name}_metrics.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to extract feature importance before saving
        if self.feature_names is not None:
            feature_importance = self.extract_feature_importance()
            if feature_importance:
                self.results['feature_importance'] = feature_importance
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Metrics saved to: {output_path}")
