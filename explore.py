#imports

import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy.stats import shapiro, spearmanr


def perform_spearmanr_test(df, col_name):
    """
    Perform a Spearman rank correlation test between the specified column and 'value' column in the DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data.
    - col_name (str): The name of the column to test for correlation with 'value'.

    Returns:
    None: Prints the result of the Spearman rank correlation test.
    """
    r, p = spearmanr(df[col_name], df['value'])
    
    if p < 0.05:
        print(f"Result: There is a significant monotonic relationship between {col_name} and value (p-value={p}).")
    else:
        print(f"Result: There is no significant monotonic relationship between {col_name} and value (p-value={p}).")



def kbest_features(df, col_name, k=2):
    """
    Selects the top k best features for regression from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing features and the target column.
        col_name (str): The name of the target column.
        k (int): The number of top features to select (default is 2).

    Returns:
        selected_df (DataFrame): A DataFrame with two columns: 'Column Name' and 'Score'.
            'Column Name' contains the column names of the selected features.

    Example:
        selected_features = select_k_best_features(your_dataframe, 'value', k=2)
    """
    # Create X and y
    X = df.drop(columns=[col_name])  # Remove the target column
    y = df[col_name]
    
    # Filter X to keep only columns that can be converted to float
    X = X.select_dtypes(include=['number', 'float'])  # Keep numeric and float columns
    
    # Initialize SelectKBest with f_regression
    skb = SelectKBest(f_regression, k=k)
    
    # Fit SelectKBest on X and y
    skb.fit(X, y)
    
    # Get the mask of selected features
    skb_mask = skb.get_support()
    
    # Get the scores for all features
    feature_scores = skb.scores_
    
    # Create a DataFrame with column names and scores
    selected_df = pd.DataFrame({
        'Kbest': X.columns[skb_mask],
        'Score': feature_scores[skb_mask]
    })
    
    # Sort the DataFrame by score in descending order
    selected_df = selected_df.sort_values(by='Score', ascending=False)
    
    selected_df.reset_index(drop=True, inplace=True)
    
    selected_df.drop('Score', axis=1, inplace=True)
    
    return selected_df

def rfe_features(df, col_name, n_features=3):
    """
    Selects a specified number of features from a DataFrame using Linear Regression and RFE.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the features and target variable.
        col_name (str): The name of the target column.
        n_features (int): The number of features to select (default is 3).

    Returns:
        pd.Series: A Series containing the selected feature names.
    """
    # Drop the target column from the DataFrame
    X = df.drop(columns=[col_name])
    
    # Select columns with numeric or float data types
    X = X.select_dtypes(include=['number', 'float'])
    
    # Extract the target variable
    y = df[col_name]
    
    # Initialize a Linear Regression model
    lm = LinearRegression()
    
    # Initialize RFE (Recursive Feature Elimination) with the specified number of features to select
    rfe = RFE(lm, n_features_to_select=n_features)
    
    # Fit RFE on the data
    rfe.fit(X, y)
    
    # Get the mask of selected features
    rfe_mask = rfe.get_support()
    
    # Get the column names of selected features
    selected_features = X.columns[rfe_mask]
    
    # Create a DataFrame to present the selected features
    selected_df = pd.DataFrame({'RFE': selected_features})
    
    return selected_df


def lasso_features(df, col_name, k=2):
    """
    Performs LASSO feature selection to select the top k features for regression from a DataFrame.

    Parameters:
        df (DataFrame): The input DataFrame containing features and the target column.
        col_name (str): The name of the target column.
        k (int): The number of top features to select (default is 2).

    Returns:
        selected_df (DataFrame): A DataFrame with two columns: 'Column Name' and 'Coefficient'.
            'Column Name' contains the column names of the selected features.
            'Coefficient' contains the corresponding LASSO regression coefficients.

    Example:
        selected_features = lasso_feature_selection(your_dataframe, 'value', k=2)
    """
    # Create X and y
    X = df.drop(columns=[col_name])  # Remove the target column
    y = df[col_name]
    
    # Filter X to keep only numeric and float columns
    X = X.select_dtypes(include=['number', 'float'])
    
    # Initialize LASSO regression with alpha=1.0 (adjust as needed)
    lasso = Lasso(alpha=1.0)
    
    # Fit LASSO on X and y
    lasso.fit(X, y)
    
    # Create a DataFrame with selected column names and their coefficients
    selected_df = pd.DataFrame({
        'Lasso': X.columns,
        'Coefficient': lasso.coef_
    })
    
    # Sort the DataFrame by absolute coefficient value in descending order
    selected_df['Coefficient'] = abs(selected_df['Coefficient'])
    selected_df = selected_df.sort_values(by='Coefficient', ascending=False)
    
    # Keep the top k features
    selected_df = selected_df.head(k)
    
    selected_df.reset_index(drop=True, inplace=True)
    
    selected_df.drop('Coefficient', axis=1, inplace=True)
    
    return selected_df


def feature_selections_results(df, col_name, k=2):
    """
    Combine the results of three feature selection functions into a final DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing features and the target column.
    col_name (str): The name of the target column.
    k (int, optional): The number of top features to select (default is 2).

    Returns:
    DataFrame: A DataFrame containing the selected features from three different feature selection methods.
    """
    selected_df1 = kbest_features(df, col_name, k)
    selected_df2 = rfe_features(df, col_name, k)
    selected_df3 = lasso_features(df, col_name, k)
    
    final_selected_df = pd.concat([selected_df1, selected_df2, selected_df3], axis=1)
    
    return final_selected_df

def MinMax_Scaler(df):
    
    mms = MinMaxScaler()

    to_scale = ['bedrooms', 'bathrooms', 'area', 'year']

    df[to_scale] = mms.fit_transform(df[to_scale])
    
    return df