# Music Recommendation System - Beats Per Minute Prediction

This project focuses on building a music recommendation system by predicting the "Beats Per Minute" (BPM) of songs based on various audio features as a submition for a Kaggle competition. The goal is to leverage machine learning techniques to accurately estimate the tempo of a track.

Competition link:
Walter Reade and Elizabeth Park. Predicting the Beats-per-Minute of Songs. https://kaggle.com/competitions/playground-series-s5e9, 2025. Kaggle.

## Project Overview

The project involves the following steps:

1.  **Data Loading and Exploration**: Loading the training and testing datasets, and performing initial data exploration to understand the structure and characteristics of the data.
2.  **Data Preprocessing**: Handling missing values, outliers, and potentially transforming features to prepare the data for model training. This includes removing outliers using the IQR method and normalizing the data.
3.  **Feature Analysis**: Analyzing the relationships between features and the target variable (BPM), including visualizing correlations.
4.  **Model Selection and Training**: Experimenting with different regression models, including:
    *   Random Forest Regressor
    *   Hist Gradient Boosting Regressor
    *   LightGBM Regressor
    *   CatBoost Regressor
    *   XGBoost Regressor
    *   KNeighbors Regressor
5.  **Hyperparameter Tuning**: Using techniques like RandomizedSearchCV and Optuna to find the optimal hyperparameters for the selected models.
6.  **Ensemble Modeling**: Creating an ensemble model using a Voting Regressor to combine the predictions of multiple models for improved performance. The ensemble is weighted based on the inverse of the RMSE scores of the individual models.
7.  **Prediction and Submission**: Making predictions on the test set and generating a submission file in the specified format.
8.  **Model Evaluation**: Evaluating the performance of the trained models using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.
9.  **Feature Importance Analysis**: Analyzing the importance of different features in predicting BPM using the trained models.
10. **Statistical Analysis**: Using OLS regression to understand the linear relationships between the features and the target variable.

## Data

The project uses two datasets: `train.csv` and `test.csv`, containing various audio features and the target variable "BeatsPerMinute" in the training set.

## Dependencies

The project requires the following libraries:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   catboost
*   xgboost
*   lightgbm
*   optuna
*   patsy
*   statsmodels

## Usage

To run the project, execute the code cells in the provided Jupyter Notebook (or Colab notebook) in sequence. The notebook includes steps for data loading, preprocessing, model training, evaluation, and prediction.

## Results

The project achieved a low RMSE on the prediction set, but the dataset appears to be too noisy or too short, not allowing the model to go much further. The ensemble model is expected to provide better performance compared to individual models.
