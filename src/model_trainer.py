import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_loader import load_data
from .eda_preprocessing import perform_eda, preprocess_data
from .feature_engineering import perform_feature_engineering
from .model_evaluator import evaluate_model_performance, plot_model_comparison, plot_predictions_vs_true, plot_residuals, plot_shap_explanations # ADDITION v5: Import plot_shap_explanations
from .utils import save_results, load_results

def train_and_evaluate_all_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
    preprocessor, poly_transformer,
    models_dir: str, results_dir: str
):
    """
    Defines, trains, tunes, and evaluates multiple regression models.
    Saves trained models and stores evaluation results.
    Assumes y_train and y_test are already log-transformed.
    Inverse transforms predictions and true values for metric calculation and plotting.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Log-transformed training target.
        y_test (pd.Series): Log-transformed testing target.
        preprocessor: The fitted ColumnTransformer for preprocessing.
        poly_transformer: The fitted PolynomialFeatures object.
        models_dir (str): Directory to save trained models.
        results_dir (str): Directory to save evaluation results and plots.
    """
    print("\n--- Starting Model Training and Evaluation ---")
    
    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save preprocessor and poly_transformer
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    joblib.dump(poly_transformer, os.path.join(models_dir, 'poly_transformer.joblib'))
    print(f"Preprocessor and PolynomialFeatures saved to {models_dir}")

    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {'fit_intercept': [True, False]}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, eval_metric='rmse', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    }

    all_model_results = {}
    best_models = {}

    for name, config in models.items():
        print(f"\n--- Training and Tuning {name} ---")
        model = config['model']
        params = config['params']

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, params, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[name] = best_model

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation R2 score for {name}: {grid_search.best_score_:.4f}")

        # Evaluate on test set
        y_pred_log = best_model.predict(X_test)
        # Inverse transform predictions back to original scale before calculating metrics
        y_pred = np.expm1(y_pred_log)

        # Inverse transform y_test for metric calculation and plotting
        y_test_original_scale = np.expm1(y_test)

        metrics = evaluate_model_performance(y_test_original_scale, y_pred, name)

        # Store results
        all_model_results[name] = {
            'best_params': grid_search.best_params_,
            'cv_r2_score': grid_search.best_score_, # This R2 is on log-transformed data
            'test_metrics': metrics # These metrics are on original scale
        }
        
        # Save the trained model
        model_filename = os.path.join(models_dir, f'{name.lower().replace(" ", "_")}_model.joblib')
        joblib.dump(best_model, model_filename)
        print(f"Trained {name} model saved to {model_filename}")

        # Plot predictions vs. true and residuals for each model
        plot_predictions_vs_true(y_test_original_scale, y_pred, name, results_dir)
        plot_residuals(y_test_original_scale, y_pred, name, results_dir)

    # Save all model results to a JSON file
    save_results(all_model_results, os.path.join(results_dir, 'model_training_results.json'))
    print(f"All model training results saved to {results_dir}/model_training_results.json")

    # Plot comparison of all models
    plot_model_comparison(all_model_results, results_dir)
    print("--- Model Training and Evaluation Complete ---")

    return best_models, all_model_results

def train_and_evaluate_advanced_models(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
    models_dir: str, results_dir: str
):
    """
    Performs advanced hyperparameter tuning for top models (XGBoost and Random Forest).
    Assumes y_train and y_test are already log-transformed.
    Inverse transforms predictions and true values for metric calculation and plotting.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Log-transformed training target.
        y_test (pd.Series): Log-transformed testing target.
        models_dir (str): Directory to save trained models.
        results_dir (str): Directory to save evaluation results and plots.
    """
    print("\n--- Starting Advanced Hyperparameter Tuning ---")

    advanced_tuning_models = {
        'XGBoost_Tuned': {
            'model': XGBRegressor(random_state=42, eval_metric='rmse', use_label_encoder=False),
            'params': {
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        },
        'RandomForest_Tuned': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [200, 300, 400],
                'max_depth': [15, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    advanced_tuning_results = {}
    best_tuned_models = {} # ADDITION v5: Store best tuned models

    for name, config in advanced_tuning_models.items():
        print(f"\n--- Advanced Tuning for {name} ---")
        model = config['model']
        params = config['params']

        grid_search = GridSearchCV(model, params, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='r2', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_tuned_models[name] = best_model # ADDITION v5: Store best tuned model
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best cross-validation R2 score for {name}: {grid_search.best_score_:.4f}")

        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log) # Inverse transform predictions

        y_test_original_scale = np.expm1(y_test) # Inverse transform y_test

        metrics = evaluate_model_performance(y_test_original_scale, y_pred, name + " (Tuned)")

        advanced_tuning_results[name] = {
            'best_params': grid_search.best_params_,
            'cv_r2_score': grid_search.best_score_, # This R2 is on log-transformed data
            'test_metrics': metrics # These metrics are on original scale
        }
        
        model_filename = os.path.join(models_dir, f'{name.lower()}_better_model.joblib')
        joblib.dump(best_model, model_filename)
        print(f"Better-tuned {name} model saved to {model_filename}")

        plot_predictions_vs_true(y_test_original_scale, y_pred, name + " (Tuned)", results_dir)
        plot_residuals(y_test_original_scale, y_pred, name + " (Tuned)", results_dir)

    save_results(advanced_tuning_results, os.path.join(results_dir, 'advanced_tuning_results.json'))
    print(f"Advanced tuning results saved to {results_dir}/advanced_tuning_results.json")
    
    print("\n--- Advanced Hyperparameter Tuning Complete ---")
    return best_tuned_models # ADDITION v5: Return best tuned models


if __name__ == "__main__":
    # Define dataset details
    DATASET_PATH = 'data/insurance.csv'
    TARGET_COLUMN = 'charges'
    NUMERICAL_FEATURES = ['age', 'bmi', 'children']
    CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
    FEATURES_FOR_ENGINEERING = ['age', 'bmi'] # Features to create polynomial/interaction terms from

    # Define directories for models and results (will be created if they don't exist)
    # CHANGE v3: Using versioned folders for models and results
    MODELS_DIR = 'models_v3'
    RESULTS_DIR = 'results_v3'

    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load Data
    df = load_data(DATASET_PATH)
    if df.empty:
        print("Exiting due to data loading error.")
    else:
        # 2. Perform EDA (on original data)
        perform_eda(df.copy(), NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, RESULTS_DIR)

        # 3. Preprocess Data (including log transform of target)
        X_processed_df, y_transformed, preprocessor = preprocess_data(df.copy(), NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN)
        
        # 4. Feature Engineering
        # Note: Feature engineering is performed on X_processed_df, not directly on df,
        # as some features (e.g., categorical) are already processed.
        X_engineered_df, poly_transformer = perform_feature_engineering(X_processed_df.copy(), FEATURES_FOR_ENGINEERING, RESULTS_DIR)

        # 5. Split Data (using log-transformed y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered_df, y_transformed, test_size=0.2, random_state=42
        )
        print(f"\nData split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

        # 6. Train and Evaluate All Models (on log-transformed target)
        best_models, all_model_results = train_and_evaluate_all_models(
            X_train, X_test, y_train, y_test,
            preprocessor, poly_transformer,
            MODELS_DIR, RESULTS_DIR
        )

        # 7. Perform Advanced Hyperparameter Tuning (on log-transformed target)
        best_tuned_models = train_and_evaluate_advanced_models( # ADDITION v5: Capture best tuned models
            X_train, X_test, y_train, y_test,
            MODELS_DIR, RESULTS_DIR
        )

        # ADDITION v5: Generate SHAP explanations for the best overall model
        # Based on previous analysis, RandomForest_Tuned (from advanced tuning) had the best MAE.
        # We'll use this model for SHAP.
        best_overall_model_name = 'RandomForest_Tuned'
        if best_overall_model_name in best_tuned_models:
            best_overall_model = best_tuned_models[best_overall_model_name]
            # Ensure X_test is a DataFrame for SHAP
            if not isinstance(X_test, pd.DataFrame):
                X_test_shap = pd.DataFrame(X_test, columns=X_engineered_df.columns)
            else:
                X_test_shap = X_test
            
            plot_shap_explanations(best_overall_model, X_test_shap, RESULTS_DIR, best_overall_model_name)
        else:
            print(f"Warning: {best_overall_model_name} not found in best_tuned_models. Skipping SHAP explanation.")


        print("\n--- Project Workflow Complete ---")
        print(f"Best models trained and saved. Evaluation results are in '{RESULTS_DIR}' directory.")
