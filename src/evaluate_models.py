import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
import joblib
import os
import xgboost as xgb
from pathlib import Path

def evaluate_all_models(X_test, y_test, models_dict, label_encoder, output_dir="evaluation_results"):
    """
    Comprehensive evaluation of all models.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = []
    
    # Binarize labels for ROC curve
    classes = label_encoder.classes_
    n_classes = len(classes)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        
        # Predictions
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
        else:
            # For XGBoost booster object if loaded directly
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(X_test)
                y_score = model.predict(dmatrix)
            else:
                y_score = None

        y_pred = model.predict(X_test) if not isinstance(model, xgb.Booster) else np.argmax(y_score, axis=1)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"{output_dir}/confusion_matrix_{name}.png")
        plt.close()
        
        # ROC Curve (One-vs-Rest)
        if y_score is not None and n_classes > 1:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot micro-average ROC curve
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"], label=f'{name} (area = {roc_auc["micro"]:0.2f})')

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Micro-averaged)')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/roc_curves.png")
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)
    print(f"Evaluation complete. Results saved to {output_dir}/")
    return metrics_df

if __name__ == "__main__":
    # This script would be called after loading test data
    # Placeholder for actual evaluation run
    print("Evaluation script initialized.")
