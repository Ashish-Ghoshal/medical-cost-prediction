import joblib
import pandas as pd
import numpy as np
import os

# Define paths to your saved models and transformers.
# These paths are relative to where you run the 'predict.py' script.
# Make sure your 'models_v3' folder is in the root of your project.
MODELS_DIR = 'models_v3'

# Define the original features used during training.
# These must match the features used in src/eda_preprocessing.py and src/feature_engineering.py
NUMERICAL_FEATURES = ['age', 'bmi', 'children']
CATEGORICAL_FEATURES = ['sex', 'smoker', 'region']
FEATURES_FOR_ENGINEERING = ['age', 'bmi'] # Features used for polynomial/interaction terms

def load_prediction_pipeline(models_dir: str):
    """
    Loads the necessary preprocessor, polynomial transformer, and the best-performing model
    for making predictions.

    Args:
        models_dir (str): Directory where the models and transformers are saved.

    Returns:
        tuple: A tuple containing the loaded preprocessor, polynomial transformer, and model.
    """
    try:
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        poly_transformer = joblib.load(os.path.join(models_dir, 'poly_transformer.joblib'))
        # Load the best performing model from the advanced tuning round
        # Based on previous analysis, RandomForest_Tuned had the best MAE after log transform.
        model = joblib.load(os.path.join(models_dir, 'randomforest_tuned_better_model.joblib'))
        print("Prediction pipeline components loaded successfully.")
        return preprocessor, poly_transformer, model
    except FileNotFoundError as e:
        print(f"Error loading pipeline components: {e}")
        print(f"Please ensure '{models_dir}' exists and contains 'preprocessor.joblib', 'poly_transformer.joblib', and 'randomforest_tuned_better_model.joblib'.")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during pipeline loading: {e}")
        return None, None, None

def preprocess_new_data(new_raw_data: pd.DataFrame, preprocessor, poly_transformer) -> pd.DataFrame:
    """
    Applies the same preprocessing and feature engineering steps to new raw data
    as were applied to the training data.

    Args:
        new_raw_data (pd.DataFrame): A DataFrame containing new, raw data for prediction.
                                     Must have the same column names as the original dataset.
        preprocessor: The fitted ColumnTransformer.
        poly_transformer: The fitted PolynomialFeatures object.

    Returns:
        pd.DataFrame: The fully preprocessed and engineered DataFrame ready for prediction.
    """
    # Apply initial preprocessing (scaling numerical, one-hot encoding categorical)
    new_data_processed = preprocessor.transform(new_raw_data)

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names_processed = NUMERICAL_FEATURES + list(ohe_feature_names)
    new_data_processed_df = pd.DataFrame(new_data_processed, columns=all_feature_names_processed, index=new_raw_data.index)

    # Apply feature engineering (polynomial and interaction terms)
    features_for_engineering_present = [f for f in FEATURES_FOR_ENGINEERING if f in new_data_processed_df.columns]
    selected_features_for_poly = new_data_processed_df[features_for_engineering_present]

    new_poly_features = poly_transformer.transform(selected_features_for_poly)
    new_poly_feature_names = poly_transformer.get_feature_names_out(features_for_engineering_present)
    new_poly_df = pd.DataFrame(new_poly_features, columns=new_poly_feature_names, index=new_raw_data.index)

    # Drop original features that were engineered and concatenate new polynomial features
    new_data_engineered_df = new_data_processed_df.drop(columns=features_for_engineering_present, errors='ignore')
    new_data_engineered_df = pd.concat([new_data_engineered_df, new_poly_df], axis=1)
    
    return new_data_engineered_df

def predict_medical_charges(new_raw_data: pd.DataFrame, preprocessor, poly_transformer, model) -> np.array:
    """
    Makes predictions on new raw data using the loaded pipeline.

    Args:
        new_raw_data (pd.DataFrame): A DataFrame containing new, raw data for prediction.
        preprocessor: The fitted ColumnTransformer.
        poly_transformer: The fitted PolynomialFeatures object.
        model: The trained machine learning model.

    Returns:
        np.array: An array of predicted medical charges in the original currency scale.
    """
    if preprocessor is None or poly_transformer is None or model is None:
        print("Prediction pipeline not fully loaded. Cannot make predictions.")
        return np.array([])

    # Preprocess and engineer features for the new data
    processed_engineered_data = preprocess_new_data(new_raw_data, preprocessor, poly_transformer)
    
    # Make prediction on the transformed data (output will be in log scale)
    predicted_charges_log = model.predict(processed_engineered_data)

    # Inverse transform the prediction back to the original scale
    predicted_charges = np.expm1(predicted_charges_log)
    
    return predicted_charges

if __name__ == "__main__":
    # Example Usage:
    # 1. Load the prediction pipeline components
    preprocessor, poly_transformer, model = load_prediction_pipeline(MODELS_DIR)

    if all([preprocessor, poly_transformer, model]):
        # 2. Create some dummy new data for prediction
        # This DataFrame must have the same column names as your original 'insurance.csv'
        example_new_data = pd.DataFrame({
            'age': [35, 28, 60],
            'sex': ['female', 'male', 'female'],
            'bmi': [27.0, 31.5, 24.8],
            'children': [0, 1, 3],
            'smoker': ['no', 'yes', 'no'],
            'region': ['southwest', 'northeast', 'southeast']
        })
        print("\nExample New Raw Data for Prediction:")
        print(example_new_data)

        # 3. Make predictions
        predicted_charges = predict_medical_charges(example_new_data, preprocessor, poly_transformer, model)

        print("\nPredicted Medical Charges for Example Data:")
        for i, charge in enumerate(predicted_charges):
            print(f"Individual {i+1}: ${charge:.2f}")
    else:
        print("\nFailed to load prediction pipeline. Please check paths and ensure models are trained.")
