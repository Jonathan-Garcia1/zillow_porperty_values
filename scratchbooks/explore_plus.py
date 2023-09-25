#imports

import pandas as pd
import numpy as np
import os

import datetime

from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from scipy.stats import shapiro, spearmanr, f_oneway, kruskal


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
        print(f"Result: There is a significant monotonic relationship between {col_name} and value (p-value={p:.4f}, r-value={r:.4f}).")
    else:
        print(f"Result: There is no significant monotonic relationship between {col_name} and value (p-value={p:.4f}, r-value={r:.4f}).")

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def perform_anova(df, category_col, value_col, alpha=0.05):
    # Extract data for each category
    categories = df[category_col].unique()
    category_df = [df[df[category_col] == category][value_col] for category in categories]

    # Perform ANOVA test
    f_statistic, p_value = f_oneway(*category_df)
    
    # Print the results
    print("Analysis of Variance (ANOVA) Test")
    print("-----------------------------------------------------")
    
    # Perform hypothesis testing
    if p_value < alpha:
        print("Result: Reject the null hypothesis, There is a significant difference in the mean", category_col, "across different counties.")
        print("F-statistic:", f'{f_statistic:.4f}')
        print("P-value:", p_value)
    else:
        print("Result: Fail to reject the null hypothesis, There is no significant difference in the mean", category_col, "across different counties.")
        print("F-statistic:", f'{f_statistic:.4f}')
        print("P-value:", p_value)

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def perform_kruskal(df, category_col, value_col, alpha=0.05):
    # Extract data for each category
    categories = df[category_col].unique()
    category_data = [df[df[category_col] == category][value_col] for category in categories]

    # Perform Kruskal-Wallis test
    h_statistic, p_value = kruskal(*category_data)
    
    # Print the results
    print("Kruskal-Wallis Test")
    print("-----------------------------------------------------")
    
    # Perform hypothesis testing
    if p_value < alpha:
        print("Result: Reject the null hypothesis, There is a significant difference in the distribution of", value_col, "across different", category_col + ".")
        print("H-statistic:", f'{h_statistic:.4f}')
        print("P-value:", p_value)
    else:
        print("Result: Fail to reject the null hypothesis, There is no significant difference in the distribution of", value_col, "across different", category_col + ".")
        print("H-statistic:", f'{h_statistic:.4f}')
        print("P-value:", p_value)

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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
    lasso = Lasso(alpha=0.5, max_iter=100000)
    
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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def MinMax_Scaler(df):
    """
    Apply Min-Max scaling to selected columns of a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the columns to be scaled.

    Returns:
        pd.DataFrame: The DataFrame with specified columns scaled using Min-Max scaling.

    Note:
        - The function applies Min-Max scaling to numeric columns (float or int) in the DataFrame.
        - The 'value' column is excluded from scaling.
        - The selected columns are scaled to the range [0, 1].
    """
    mms = MinMaxScaler()

    # Select columns to scale (excluding 'value')
    to_scale = df.select_dtypes(include=['float', 'int']).columns.tolist()
    to_scale.remove('value')

    # Apply Min-Max scaling to the selected columns
    df[to_scale] = mms.fit_transform(df[to_scale])
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def hot_encode(df):
    """
    Perform one-hot encoding on a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing categorical columns to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns converted to one-hot encoded columns.

    Note:
        - The function uses pd.get_dummies to perform one-hot encoding on the specified DataFrame.
        - All categorical columns are one-hot encoded without dropping the first category (drop_first=False).
    """
    # Use pd.get_dummies to one-hot encode the DataFrame
    df = pd.get_dummies(df, drop_first=False)
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def simple_features(df):
    """
    Create custom features for the given DataFrame.

    This function adds the following features to the DataFrame:
    - 'total_rooms': Total number of rooms, calculated as the sum of bedrooms and bathrooms.
    - 'age': Age of each entry, calculated as the difference between the current year and the 'year' column.
    - 'bdrm_area_ratio': Bedroom-to-area ratio, calculated as 'area' divided by 'bedrooms'.
    - 'bath_bdrm_ratio': Bathroom-to-bedroom ratio, calculated as 'bathrooms' divided by 'bedrooms'.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the necessary columns.

    Returns:
    pd.DataFrame: A DataFrame with the additional features added.
    """

    # Calculate total_rooms by summing bedrooms and bathrooms
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    # Get the current year
    current_year = datetime.datetime.now().year

    # Calculate the age of each entry based on the 'year' column
    df['age'] = current_year - df['year']

    # Calculate the bedroom-to-area ratio
    df['bdrm_area_ratio'] = df['area'] / df['bedrooms']

    # Calculate the bathroom-to-bedroom ratio
    df['bath_bdrm_ratio'] = df['bathrooms'] / df['bedrooms']

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def categorize_col(df, target_column):
    """
    Categorize values in a DataFrame column based on percentiles.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the target column to be categorized.
        target_column (str): The name of the column to be categorized.

    Returns:
        pd.DataFrame: The DataFrame with a new categorical column based on percentiles.

    Note:
        - The function calculates percentiles for the specified target column.
        - It defines bins and labels for categorization based on percentiles.
        - A new categorical column is created in the DataFrame to represent the categorization.
    """
    # Find percentiles for the target column
    percentiles = df[target_column].quantile([0.01, 0.05, 0.10, 0.20, 0.25, 0.75, 0.80, 0.90, 0.95, 0.99])
    
    # Define the bins and labels
    bins = [0] + list(percentiles) + [df[target_column].max()]
    labels = ['bottom_1%', 'bottom_5%', 'bottom_10%', 'bottom_20%', 'bottom_25%',
              'middle_50%', 'top_25%', 'top_20%', 'top_10%', 'top_5%', 'top_1%']

    # Create a new column based on the percentiles
    df[f'group_{target_column}'] = pd.cut(df[target_column], bins=bins, labels=labels)
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def age_group(df):
    """
    Create an 'age_group' column in the DataFrame based on age categories.

    This function adds an 'age_group' column to the DataFrame, categorizing entries based on their 'age' values
    into predefined age groups.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'age' column.

    Returns:
    pd.DataFrame: The DataFrame with the 'age_group' column added.
    """

    # Define the age group categories and labels
    age_bins = [-1, 4, 9, 19, 39, 59, 79, 200]
    age_labels = ['newest', 'very_new', 'new', 'mid', 'old', 'very_old', 'oldest']

    # Create the 'age_group' column based on the age bins and labels
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def size_group(df):
    """
    Create a 'size_group' column in the DataFrame based on area percentiles.

    This function adds a 'size_group' column to the DataFrame, categorizing entries based on their 'area' values
    into predefined size groups defined by percentiles.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'area' column.

    Returns:
    pd.DataFrame: The DataFrame with the 'size_group' column added.
    """
    # Calculate the percentiles
    percentiles = df['area'].quantile([0.01, 0.05, 0.32, 0.68, 0.95, 0.99])

    # Define the categories
    df['size_group'] = pd.cut(
        df['area'],
        bins=[0, percentiles[0.01], percentiles[0.05], percentiles[0.32], percentiles[0.68], percentiles[0.95], percentiles[0.99], df['area'].max()],
        labels=['smallest', 'very_small', 'small', 'mid', 'large', 'very_large', 'largest']
    )
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def features(df):
    """
    Create and add multiple features to the DataFrame.

    This function applies three separate feature creation functions to the input DataFrame:
    - simple_features: Adds basic features like 'total_rooms', 'age', 'bdrm_area_ratio', and 'bath_bdrm_ratio'.
    - age_group: Adds an 'age_group' column based on predefined age categories.
    - size_group: Adds a 'size_group' column based on area percentiles.

    Parameters:
    df (pd.DataFrame): The original DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with the added features.
    """
    df = simple_features(df)
    df = categorize_col(df, 'area')
    df = categorize_col(df, 'age')
    # df = age_group(df)
    # df = size_group(df)
    
    return df