#imports

import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectKBest, f_regression, RFE

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
        selected_columns (list): A list of column names corresponding to the selected features.

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
    
    # Get the column names of the selected features
    selected_columns = X.columns[skb_mask]
    
    return selected_columns