import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import shap 

def evaluate_model_performance(y_true: pd.Series, y_pred: np.array, model_name: str) -> dict:
    """
    Calculates and prints key evaluation metrics for a regression model.
    Assumes y_true and y_pred are on the original scale (after inverse transformation if applicable).

    Args:
        y_true (pd.Series): The true values of the target variable (original scale).
        y_pred (np.array): The predicted values from the model (original scale).
        model_name (str): The name of the model for display purposes.

    Returns:
        dict: A dictionary containing MAE, MSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- Evaluation for {model_name} ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")

    return {'MAE': mae, 'MSE': mse, 'R2': r2}

def plot_predictions_vs_true(y_true: pd.Series, y_pred: np.array, model_name: str, results_dir: str):
    """
    Generates a scatter plot of true vs. predicted values.
    Assumes y_true and y_pred are on the original scale.

    Args:
        y_true (pd.Series): The true values of the target variable (original scale).
        y_pred (np.array): The predicted values from the model (original scale).
        model_name (str): The name of the model for the plot title.
        results_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'True vs. Predicted Values ({model_name})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")}_predictions_vs_true.png'))
    plt.close()
    print(f"Saved True vs. Predicted plot for {model_name} to {results_dir}")

def plot_residuals(y_true: pd.Series, y_pred: np.array, model_name: str, results_dir: str):
    """
    Generates a residual plot.
    Assumes y_true and y_pred are on the original scale.

    Args:
        y_true (pd.Series): The true values of the target variable (original scale).
        y_pred (np.array): The predicted values from the model (original scale).
        model_name (str): The name of the model for the plot title.
        results_dir (str): Directory to save the plot.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.title(f'Residual Plot ({model_name})')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")}_residuals.png'))
    plt.close()
    print(f"Saved Residual plot for {model_name} to {results_dir}")

def plot_model_comparison(all_model_results: dict, results_dir: str):
    """
    Generates a bar plot comparing R2 scores of all trained models.
    Assumes R2 scores in all_model_results are on the original scale.

    Args:
        all_model_results (dict): Dictionary containing results for all models.
        results_dir (str): Directory to save the plot.
    """
    model_names = []
    r2_scores = []
    for model_name, results in all_model_results.items():
        model_names.append(model_name)
        r2_scores.append(results['test_metrics']['R2'])

    df_comparison = pd.DataFrame({'Model': model_names, 'R2 Score': r2_scores})
    df_comparison = df_comparison.sort_values(by='R2 Score', ascending=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(x='R2 Score', y='Model', data=df_comparison, palette='viridis')
    plt.title('Comparison of Model R2 Scores on Test Data (Original Scale)')
    plt.xlabel('R2 Score')
    plt.ylabel('Model')
    plt.xlim(0, 1) 
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_r2_comparison.png'))
    plt.close()
    print(f"Saved Model R2 Comparison plot to {results_dir}/model_r2_comparison.png")


def plot_shap_explanations(model, X_data: pd.DataFrame, results_dir: str, model_name: str):
    """
    Generates SHAP summary plot and dependence plots for a given model.

    Args:
        model: The trained machine learning model.
        X_data (pd.DataFrame): The feature DataFrame used for explanations (e.g., X_test).
        results_dir (str): Directory to save SHAP plots.
        model_name (str): Name of the model for plot titles.
    """
    print(f"\n--- Generating SHAP explanations for {model_name} ---")
    try:
        # Use TreeExplainer for tree-based models (Random Forest, XGBoost)
        # For other models, shap.KernelExplainer or shap.DeepExplainer might be needed.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data)

        # SHAP Summary Plot (Feature Importance)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_data, show=False)
        plt.title(f'SHAP Feature Importance ({model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_shap_summary.png'))
        plt.close()
        print(f"Saved SHAP summary plot for {model_name} to {results_dir}")

        
        if isinstance(shap_values, list): # For multi-output models like some tree-based
            avg_abs_shap_values = np.abs(shap_values[0]).mean(0)
        else:
            avg_abs_shap_values = np.abs(shap_values).mean(0)
        
        # Ensure that avg_abs_shap_values has the same length as X_data.columns
        if len(avg_abs_shap_values) != len(X_data.columns):
            print("Warning: SHAP values and feature names mismatch. Skipping dependence plot.")
        else:
            most_important_feature_idx = np.argmax(avg_abs_shap_values)
            most_important_feature_name = X_data.columns[most_important_feature_idx]
            
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                most_important_feature_name, shap_values, X_data,
                interaction_index=None, # Set to None for just the feature's effect
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {most_important_feature_name} ({model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'{model_name.lower().replace(" ", "_")}_{most_important_feature_name}_shap_dependence.png'))
            plt.close()
            print(f"Saved SHAP dependence plot for {most_important_feature_name} to {results_dir}")

    except Exception as e:
        print(f"Error generating SHAP explanations for {model_name}: {e}")
    print("--- SHAP explanation generation complete ---")
