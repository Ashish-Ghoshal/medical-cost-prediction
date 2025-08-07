import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io # FIX: Import the io module for StringIO
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def perform_eda(df: pd.DataFrame, numerical_cols: list, categorical_cols: list, target_col: str, results_dir: str):
    """
    Performs Exploratory Data Analysis (EDA) on the DataFrame and saves insights.
    This function operates on the raw, untransformed data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        target_col (str): Name of the target column.
        results_dir (str): Directory to save EDA plots and insights.
    """
    print("\n--- Starting EDA ---")
    eda_insights = {}

    # Basic information
    print("DataFrame Info:")
    # FIX: Correctly capture df.info() output using io.StringIO
    buffer = io.StringIO()
    df.info(verbose=True, show_counts=True, buf=buffer)
    eda_insights['info'] = buffer.getvalue()
    print(eda_insights['info']) # Print it as it was before

    print("\nDescriptive Statistics:")
    print(df.describe())
    eda_insights['descriptive_statistics'] = df.describe().to_dict()

    # Missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    eda_insights['missing_values'] = missing_values[missing_values > 0].to_dict()

    # Distributions of numerical features
    print("\nPlotting numerical feature distributions...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols):
        plt.subplot(2, int(np.ceil(len(numerical_cols)/2)), i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/numerical_distributions.png')
    plt.close()
    print(f"Saved numerical distributions to {results_dir}/numerical_distributions.png")

    # Distributions of categorical features
    print("Plotting categorical feature distributions...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols):
        plt.subplot(2, int(np.ceil(len(categorical_cols)/2)), i + 1)
        sns.countplot(y=df[col], palette='viridis')
        plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/categorical_distributions.png')
    plt.close()
    print(f"Saved categorical distributions to {results_dir}/categorical_distributions.png")

    # Correlation heatmap for numerical features
    print("Plotting correlation heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols + [target_col]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig(f'{results_dir}/correlation_heatmap.png')
    plt.close()
    print(f"Saved correlation heatmap to {results_dir}/correlation_heatmap.png")

    # Relationship between categorical features and target
    print("Plotting categorical features vs. target...")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_cols):
        plt.subplot(2, int(np.ceil(len(categorical_cols)/2)), i + 1)
        sns.boxplot(x=col, y=target_col, data=df, palette='pastel')
        plt.title(f'{col} vs. {target_col}')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/categorical_vs_target.png')
    plt.close()
    print(f"Saved categorical vs. target plots to {results_dir}/categorical_vs_target.png")

    # Save EDA insights to a JSON file
    with open(f'{results_dir}/eda_insights.json', 'w') as f:
        json.dump(eda_insights, f, indent=4)
    print(f"Saved EDA insights to {results_dir}/eda_insights.json")
    print("--- EDA Complete ---")


def preprocess_data(df: pd.DataFrame, numerical_cols: list, categorical_cols: list, target_col: str):
    """
    Handles missing values, encodes categorical features, and scales numerical features.
    Applies log1p transformation to the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        target_col (str): Name of the target column.

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: The preprocessed features (X).
               - pd.Series: The log-transformed target variable (y).
               - ColumnTransformer: The fitted preprocessor object.
    """
    print("\n--- Starting Data Preprocessing ---")
    # Drop rows with any missing values for simplicity in this example
    # For real-world, consider imputation strategies
    initial_rows = df.shape[0]
    df = df.dropna()
    if df.shape[0] < initial_rows:
        print(f"Dropped {initial_rows - df.shape[0]} rows due to missing values.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # CHANGE v3: Apply log transformation to the target variable 'y'
    # np.log1p(x) computes log(1+x), which is safer than log(x) if x can be 0 or very small.
    print(f"Applying log transformation to the target variable '{target_col}'...")
    y = np.log1p(y)
    print(f"Target variable '{target_col}' transformed. First 5 values: {y.head().tolist()}")

    # Create a column transformer for preprocessing
    # Numerical features will be scaled
    # Categorical features will be one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(ohe_feature_names) # Assuming no 'remainder' columns are added back for simplicity here

    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

    print("Data preprocessing complete.")
    return X_processed_df, y, preprocessor
