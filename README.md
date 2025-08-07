# Medical Cost Prediction with Advanced Regression Models

# 

This repository presents a comprehensive machine learning project aimed at predicting individual medical costs based on various health and demographic attributes. It showcases a complete end-to-end workflow, from extensive Exploratory Data Analysis (EDA) and robust preprocessing to training and evaluating multiple advanced regression models, including hyperparameter tuning and model persistence.

## 📝 Table of Contents

# 

1.  [🌟 Features](https://www.google.com/search?q=%23-features "null")
    
2.  [🛠️ Technologies and Libraries](https://www.google.com/search?q=%23-technologies-and-libraries "null")
    
3.  [📂 Project Structure](https://www.google.com/search?q=%23-project-structure "null")
    
4.  [🚀 Setup and Installation](https://www.google.com/search?q=%23-setup-and-installation "null")
    
5.  [🧠 Usage](https://www.google.com/search?q=%23-usage "null")
    
    *   [Running the Project Locally](https://www.google.com/search?q=%23running-the-project-locally "null")
        
    *   [Interpreting Results](https://www.google.com/search?q=%23interpreting-results "null")
        
    *   [Using Pre-trained Models for Prediction](https://www.google.com/search?q=%23using-pre-trained-models-for-prediction "null")
        
6.  [📊 Performance Summary](https://www.google.com/search?q=%23-performance-summary "null")
    
7.  [🔍 Model Interpretability (SHAP)](https://www.google.com/search?q=%23-model-interpretability-shap "null")
    
8.  [🚀 Future Enhancements](https://www.google.com/search?q=%23-future-enhancements "null")
    
9.  [🤝 Contributing](https://www.google.com/search?q=%23-contributing "null")
    
10.  [📜 License](https://www.google.com/search?q=%23-license "null")
    

## 🌟 Features

# 

*   **Thorough Exploratory Data Analysis (EDA)**: In-depth analysis of dataset characteristics, distributions, relationships, and potential issues like outliers or skewness.
    
*   **Robust Data Preprocessing**: Handles categorical features using one-hot encoding, scales numerical features, and applies a **log transformation to the target variable** (`charges`) to address its skewed distribution.
    
*   **Intelligent Feature Engineering**: Creates new, meaningful features (polynomial and interaction terms) from existing ones to improve model performance.
    
*   **Advanced Model Training**: Implements and trains a suite of regression models:
    
    *   **Linear Regression**
        
    *   **Random Forest Regressor**
        
    *   **XGBoost Regressor**
        
    *   **Support Vector Regressor (SVR)**
        
    *   **K-Nearest Neighbors (KNN) Regressor**
        
*   **Systematic Hyperparameter Tuning**: Utilizes `GridSearchCV` to optimize model performance by finding the best hyperparameters for each algorithm.
    
*   **Comprehensive Model Evaluation**: Assesses model performance using key metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **score**, reported on the original scale of medical charges.
    
*   **Comparative Visualization**: Generates insightful plots to compare the performance of different models on both training and testing data.
    
*   **Model Persistence**: Saves trained models, scalers, and preprocessors to disk using `joblib` for easy loading and reuse.
    
*   **Experiment Tracking**: Stores results of EDA, feature engineering, and model training (including best hyperparameters and metrics) in a structured format for easy analysis and iteration.
    
*   **Model Interpretability (SHAP)**: Provides insights into model predictions using SHAP (SHapley Additive exPlanations) values, helping to understand feature importance and impact.
    
*   **Prediction Pipeline**: A dedicated script (`predict.py`) to easily load the best-trained model and make predictions on new, unseen data.
    

## 🛠️ Technologies and Libraries

# 

*   **Python**: The core programming language.
    
*   **Conda**: For environment management.
    
*   **Pandas**: For efficient data manipulation and analysis.
    
*   **NumPy**: For numerical operations.
    
*   **Scikit-learn**: Extensive library for machine learning, including preprocessing, model selection, training, and evaluation.
    
*   **XGBoost**: Optimized distributed gradient boosting library.
    
*   **Matplotlib** and **Seaborn**: For creating high-quality static, animated, and interactive visualizations.
    
*   **Joblib**: For saving and loading Python objects, especially large NumPy arrays and scikit-learn models.
    
*   **SHAP**: For model interpretability.
    

## 📂 Project Structure

# 

    .
    ├── data/
    │   └── insurance.csv
    ├── models_v3/
    │   ├── linear_regression_model.joblib
    │   ├── random_forest_model.joblib
    │   ├── xgboost_model.joblib
    │   ├── svr_model.joblib
    │   ├── knn_model.joblib
    │   ├── preprocessor.joblib
    │   ├── poly_transformer.joblib
    │   ├── xgboost_tuned_better_model.joblib # From advanced tuning
    │   └── randomforest_tuned_better_model.joblib # From advanced tuning
    ├── results_v3/
    │   ├── eda_insights.json
    │   ├── feature_engineering_insights.json
    │   ├── model_training_results.json
    │   ├── advanced_tuning_results.json
    │   ├── *.png (various plots including SHAP plots)
    ├── experiments/
    │   └── medical_cost_prediction.ipynb  (Google Colab notebook for reference)
    ├── README.md
    ├── requirements.txt
    └── src/
        ├── __init__.py
        ├── data_loader.py
        ├── eda_preprocessing.py
        ├── feature_engineering.py
        ├── model_trainer.py
        ├── model_evaluator.py
        ├── predict.py
        └── utils.py
    
    

*   `data/`: Contains the raw dataset (`insurance.csv`).
    
*   `models_v3/`: This directory will store all trained machine learning models, data scalers, and preprocessors from the latest run (with log transformation).
    
*   `results_v3/`: This directory will store all evaluation results (JSON files) and generated plots (`.png` files) from the latest run, including SHAP plots.
    
*   `experiments/`: Houses the Google Colab notebook (`medical_cost_prediction.ipynb`), which provides a self-contained, end-to-end execution of the entire project workflow, useful for cloud-based training.
    
*   `src/`: Contains modular Python scripts for different components of the project:
    
    *   `data_loader.py`: Handles loading the dataset.
        
    *   `eda_preprocessing.py`: Functions for EDA, data cleaning, and preprocessing (encoding, scaling).
        
    *   `feature_engineering.py`: Functions for creating new features.
        
    *   `model_trainer.py`: Defines, trains, and tunes different regression models. This is the main script to run.
        
    *   `model_evaluator.py`: Calculates metrics, generates plots for model evaluation and comparison, and now includes SHAP explanation plots.
        
    *   `predict.py`: A script for making predictions using the saved models.
        
    *   `utils.py`: Utility functions for saving and loading results and objects.
        
*   `README.md`: This file, providing an overview of the project.
    
*   `requirements.txt`: Lists all Python dependencies.
    

## 🚀 Setup and Installation

# 

1.  **Clone the Repository:**
    
        git clone https://github.com/your_username/medical-cost-prediction.git
        cd medical-cost-prediction
        
    
2.  **Download the Dataset:**
    
    *   The dataset used is the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance "null").
        
    *   Download `insurance.csv` and place it in the `data/` directory within your cloned repository.
        
3.  **Create and Initialize Conda Environment:**
    
    First, initialize Conda for your shell (if you haven't already). This is a one-time setup.
    
        conda init bash # For Git Bash
        # After running this, close and reopen your Git Bash terminal.
        
    
    Then, create and activate your environment:
    
        conda create -n m_c_venv python=3.9 # Or your preferred Python version
        conda activate m_c_venv
        
    
4.  **Install Required Libraries:**
    
        pip install -r requirements.txt
        
    
5.  **Download Pre-trained Models and Results (Optional, but Recommended for Quick Use):**
    
    *   If you have already run the Google Colab notebook, you will have `models_v3/` and `results_v3/` folders in your Google Drive (from your `/content/drive/MyDrive/elevate_proj/task_3/` path).
        
    *   **Download the entire contents** of these `models_v3` and `results_v3` folders from your Google Drive.
        
    *   Place the downloaded `models_v3` folder and `results_v3` folder directly into the root of your local `medical-cost-prediction` repository (i.e., alongside `data/`, `src/`, `README.md`, etc.). This will allow the `model_trainer.py` script to load and use these pre-trained artifacts, and the `predict.py` script to function immediately.
        

## 🧠 Usage

### Running the Project Locally

# 

The `src/model_trainer.py` script orchestrates the entire machine learning workflow.

1.  Ensure your Conda environment `m_c_venv` is activated (`conda activate m_c_venv`).
    
2.  Ensure the `insurance.csv` dataset is in the `data/` directory.
    
3.  Run the main training script:
    
        python -m src.model_trainer
        
    
    This script will perform:
    
    *   Data loading and initial preprocessing.
        
    *   Exploratory Data Analysis (EDA) and save plots/insights to `results_v3/`.
        
    *   Feature engineering (including log transformation of target) and save insights to `results_v3/`.
        
    *   Splitting data into training and testing sets.
        
    *   Training and hyperparameter tuning of all five regression models.
        
    *   Evaluation of each model on the test set.
        
    *   Saving the best-performing models, preprocessor, and polynomial transformer to `models_v3/`.
        
    *   Saving all model evaluation results and comparison plots to `results_v3/`.
        
    *   **Generating SHAP explanation plots** for the best-performing model (Random Forest Tuned).
        

### Interpreting Results

# 

The project generates various outputs to help you understand the models' performance:

*   **EDA Plots** (`results_v3/*.png`): Histograms, box plots, and correlation heatmaps provide insights into data distributions and relationships.
    
*   **Evaluation Metrics** (printed to console and in `results_v3/model_training_results.json`, `results_v3/advanced_tuning_results.json`): MAE, MSE, and R2 scores are reported for each model, indicating prediction accuracy and goodness of fit.
    
    *   **MAE**: Average absolute difference between predictions and actual values. Lower values are better.
        
    *   **MSE**: Average of the squared differences between predictions and actual values (penalizes larger errors more). Lower values are better.
        
    *   R2 **Score**: Proportion of variance in the dependent variable that is predictable from the independent variable(s). An R2 score closer to 1 signifies a better-fitting model.
        
*   **Comparison Plots** (`results_v3/model_r2_comparison.png`): Visualizations comparing the R2 scores of different models help in quickly identifying the best-performing model.
    
*   **Residual Plots** (`results_v3/*_residuals.png`): Show the distribution of errors, helping to identify potential issues like heteroscedasticity or non-linearity.
    
*   **SHAP Plots** (`results_v3/*_shap_summary.png`, `results_v3/*_shap_dependence.png`): These plots provide insights into how each feature contributes to the model's predictions.
    
    *   **SHAP Summary Plot**: Shows the overall feature importance and the impact (positive or negative) of each feature on the predictions.
        
    *   **SHAP Dependence Plot**: Illustrates how a single feature's value affects the prediction, often showing interactions with other features.
        

### Using Pre-trained Models for Prediction

# 

The `src/predict.py` script provides a dedicated pipeline to make predictions on new data.

1.  Ensure your Conda environment `m_c_venv` is activated (`conda activate m_c_venv`).
    
2.  Ensure you have the `models_v3/` folder (containing `preprocessor.joblib`, `poly_transformer.joblib`, and your chosen `randomforest_tuned_better_model.joblib`) in the root of your project.
    
3.  Run the prediction script:
    
        python src/predict.py
        
    
    This script will:
    
    *   Load the necessary preprocessor, polynomial transformer, and the best-trained model (`randomforest_tuned_better_model.joblib`).
        
    *   Use example new data (defined within the script) to demonstrate a prediction.
        
    *   Print the predicted medical charges.
        
    
    You can modify the `example_new_data` DataFrame in `src/predict.py` to test with your own new data points.
    

## 📊 Performance Summary

# 

This section summarizes the key findings from the model training and evaluation, highlighting the impact of different approaches.

### Initial Models (No Log Transform)

| Model | Test R2 Score | Test MAE | Test MSE |
| --- | --- | --- | --- |
| XGBoost | 0.8815 | 2440.73 | 18394316.51 |
| Random Forest | 0.8644 | 2567.40 | 21046967.04 |
| Linear Regression | 0.7809 | 4247.78 | 34015745.99 |
| KNeighbors | 0.6724 | 4152.06 | 50857677.33 |
| SVR | -0.0474 | 8365.84 | 162603263.13 |


### Advanced Tuning (No Log Transform)

| Model | Test R2 Score | Test MAE | Test MSE |
| --- | --- | --- | --- |
| XGBoost_Tuned | 0.8808 | 2499.39 | 18499485.48 |
| RandomForest_Tuned | 0.8743 | 2489.81 | 19511039.96 |

### Initial Models (With Log Transform on Target)

| Model | Test R2 Score | Test MAE | Test MSE |
| --- | --- | --- | --- |
| Random Forest | 0.8749 | 2093.83 | 19429221.10 |
| XGBoost | 0.8748 | 1997.87 | 19432215.63 |
| SVR | 0.7704 | 2527.63 | 35642934.94 |
| Linear Regression | 0.6269 | 3872.38 | 57915586.65 |
| KNeighbors | 0.6253 | 3952.88 | 58174610.12 |

### Advanced Tuning (With Log Transform on Target)

| Model | Test R2 Score | Test MAE | Test MSE |
| --- | --- | --- | --- |
| RandomForest_Tuned | 0.8782 | 2020.78 | 18901689.02 |
| XGBoost_Tuned | 0.8700 | 2059.46 | 20187682.58 |

**Key Takeaways:**

*   The **log transformation significantly improved the Mean Absolute Error (MAE)** for tree-based models (XGBoost and Random Forest), indicating that predictions are, on average, closer to the true values in dollar amounts.
    
*   The **`RandomForest_Tuned` model with log transformation achieved the best MAE** (approx. \\$2020), making it the most accurate in terms of average absolute prediction error. Its R2 score is also highly competitive.
    
*   SVR's performance drastically improved with the log transformation, becoming a viable model.
    

## 🚀 Future Enhancements

# 

*   **Automated Feature Selection**: Implement more advanced feature selection techniques (e.g., Recursive Feature Elimination, SelectKBest with different statistical tests) to automatically identify the most impactful features.
    
*   **Ensemble Modeling**: Explore stacking or blending techniques to combine predictions from multiple models, potentially achieving even higher accuracy than individual models.
    
*   **Deep Learning Models**: Investigate the use of neural networks (e.g., using TensorFlow or PyTorch) for regression tasks, especially if the dataset size grows or non-linear relationships are complex.
    
*   **Model Deployment**: Create a simple web API (e.g., using Flask or FastAPI) to serve the best-performing model, allowing real-time predictions via HTTP requests. This would make the model accessible for integration into other applications.
    
*   **Continuous Integration/Continuous Deployment (CI/CD)**: Set up CI/CD pipelines to automate testing, model retraining, and deployment whenever changes are pushed to the repository, ensuring the model remains up-to-date and performant.
    
*   **Dockerization**: Containerize the application using Docker to ensure consistent environments across development, testing, and production, simplifying deployment and scalability.
    
*   **More In-depth XAI**: Explore additional SHAP visualization types (e.g., force plots for individual predictions) or other Explainable AI techniques to provide even deeper insights into model behavior.
    

## 🤝 Contributing

# 

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

## 📜 License

# 

This project is licensed under the MIT License.