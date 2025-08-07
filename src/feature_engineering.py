import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import json

def perform_feature_engineering(df: pd.DataFrame, features_to_engineer: list, results_dir: str):
    """
    Performs feature engineering by creating polynomial and interaction features.

    Args:
        df (pd.DataFrame): The input DataFrame (after initial preprocessing, before target split).
        features_to_engineer (list): List of numerical features to use for engineering.
        results_dir (str): Directory to save feature engineering insights.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: DataFrame with new engineered features.
               - PolynomialFeatures: The fitted PolynomialFeatures object.
    """
    print("\n--- Starting Feature Engineering ---")
    fe_insights = {}

    # Select only the features that are numerical and relevant for engineering
    # Ensure features_to_engineer are present in the DataFrame's columns
    selected_features_df = df[features_to_engineer]

    # Create Polynomial Features (degree 2 for simplicity)
    # This will create polynomial features (e.g., feature^2) and interaction terms (e.g., feature1 * feature2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    
    poly_features = poly.fit_transform(selected_features_df)
    poly_feature_names = poly.get_feature_names_out(features_to_engineer)

    # Create a DataFrame for the new polynomial features
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # Drop original features that are now represented by polynomial features to avoid multicollinearity
    # and then concatenate with the new features.
    # We only drop them if they are part of the `features_to_engineer` list.
    df_engineered = df.drop(columns=features_to_engineer, errors='ignore').copy()
    df_engineered = pd.concat([df_engineered, poly_df], axis=1)

    fe_insights['engineered_features_count'] = len(poly_feature_names)
    fe_insights['engineered_feature_names'] = list(poly_feature_names)
    print(f"Created {fe_insights['engineered_features_count']} new features.")
    print(f"New DataFrame shape: {df_engineered.shape}")

    # Save feature engineering insights to a JSON file
    with open(f'{results_dir}/feature_engineering_insights.json', 'w') as f:
        json.dump(fe_insights, f, indent=4)
    print(f"Saved feature engineering insights to {results_dir}/feature_engineering_insights.json")
    print("--- Feature Engineering Complete ---")

    return df_engineered, poly
