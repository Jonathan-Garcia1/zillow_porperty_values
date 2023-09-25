import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from math import sqrt


def split_train_val_test(df):
    """
    Split the input DataFrame into training, validation, and test sets.
    
    Parameters:
    df (DataFrame): The input DataFrame containing the data to be split.
    
    Returns:
    train (DataFrame): The training dataset.
    val (DataFrame): The validation dataset.
    test (DataFrame): The test dataset.
    """
    # Set a random seed for reproducibility
    seed = 42
    
    # Split the data into initial training and the combination of validation and test sets
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    
    # Further split the validation and test sets
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test


def scale_train_val_test(train, val, test):
    """
    Scale specified columns in the training, validation, and test datasets using Min-Max scaling.
    
    Parameters:
    train (DataFrame): The training dataset.
    val (DataFrame): The validation dataset.
    test (DataFrame): The test dataset.
    
    Returns:
    train (DataFrame): Scaled training dataset.
    val (DataFrame): Scaled validation dataset.
    test (DataFrame): Scaled test dataset.
    """
    # Initialize the MinMaxScaler
    mms = MinMaxScaler()
    
    # Define columns to be scaled
    columns_to_scale = ['bedrooms', 'bathrooms', 'area', 'year']
    
    # Fit the scaler on the training data for specified columns
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test



def split_scale_tvt(df):
    """
    Split the input DataFrame and scale it using Min-Max scaling.
    
    Parameters:
    df (DataFrame): The input DataFrame containing the data to be split and scaled.
    
    Returns:
    train (DataFrame): Scaled training dataset.
    val (DataFrame): Scaled validation dataset.
    test (DataFrame): Scaled test dataset.
    """
    train, val, test = split_train_val_test(df)
    train, val, test = scale_train_val_test(train, val, test)
    return train, val, test

def xy_split(df):
    """
    Split the input DataFrame into feature matrix (X) and target vector (y).
    
    Parameters:
    df (DataFrame): The input DataFrame containing features and target.
    
    Returns:
    X (DataFrame): Feature matrix (all columns except 'value').
    y (Series): Target vector (column 'value').
    """
    # Split the dataset into feature columns (X) and target column (y)
    return df.drop(columns=['value']), df.value

def data_pipeline(df):
    df.drop(columns=['county','state'], inplace=True)
    train, val, test = split_scale_tvt(df)
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return train, val, test, X_train, y_train, X_val, y_val, X_test, y_test

# def build_baselines(y_train):
#     baselines = pd.DataFrame({'y_actual': y_train,
#                           'y_mean': y_train.mean()})
#     return baselines


def eval_baseline(y_train):
    """
    Calculate the Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Parameters:
    y_train.
    
    Returns:
    rmse (float): Root Mean Squared Error (RMSE) between y_actual and y_hat.
    """
    baselines = pd.DataFrame({'y_actual': y_train,
                          'y_mean': y_train.mean()})
    
    return sqrt(mean_squared_error(baselines.y_actual, baselines.y_mean))

def eval_model(y_actual, y_hat):
    
    """Calculate the RMSE.
    
       Pass in the actual values first and the predicted values second."""
    
    return sqrt(mean_squared_error(y_actual, y_hat))



def update_model_results(model_name, train_rmse, val_rmse, model_results=None):
    """
    Update the model results DataFrame with the model name and RMSE values.

    Parameters:
    model_name (str): The name of the model.
    train_rmse (float): RMSE value for the training set.
    val_rmse (float): RMSE value for the validation set.
    model_results (DataFrame, optional): Existing DataFrame to store results.

    Returns:
    model_results: Updated DataFrame containing model name and RMSE results.
    """
    # Create a DataFrame with model name and RMSE results
    results_df = pd.DataFrame({
        'Model': [model_name],
        'Train_RMSE': [train_rmse],
        'Val_RMSE': [val_rmse]
    })
    
    # Check if model_results already exists
    if model_results is not None:
        # Concatenate results with existing DataFrame
        model_results = pd.concat([model_results, results_df], ignore_index=True)
    else:
        # Create a new DataFrame if it doesn't exist
        model_results = results_df

    return model_results

def train_model(model_name, X_train, y_train, X_val, y_val, model_results=None):
    """
    Train a machine learning model and evaluate its performance on both training and validation sets.

    Parameters:
    model_name: A function that returns an instance of the machine learning model.
    X_train (DataFrame): Feature matrix for training data.
    y_train (Series): Target vector for training data.
    X_val (DataFrame): Feature matrix for validation data.
    y_val (Series): Target vector for validation data.
    model_results (DataFrame, optional): Existing DataFrame to store results.

    Returns:
    model: Trained machine learning model.
    model_results: Updated DataFrame containing model name and RMSE results.
    """
    # Fit the model on the training data
    model = model_name()
    model.fit(X_train, y_train)
    
    # Make predictions on the training set
    train_preds = model.predict(X_train)
    
    # Calculate RMSE on the training set
    train_rmse = eval_model(y_train, train_preds)
    
    # Make predictions on the validation set
    val_preds = model.predict(X_val)
    
    # Calculate RMSE on the validation set
    val_rmse = eval_model(y_val, val_preds)
    
    # Print RMSE values for training and validation sets (formatted)
    train_rmse_formatted = "${:,.2f}".format(train_rmse)
    val_rmse_formatted = "${:,.2f}".format(val_rmse)
    print(f'The train RMSE is {train_rmse_formatted}.')
    print(f'The validate RMSE is {val_rmse_formatted}.')
    
    # Extract the name of the model class without the module path
    model_name = model.__class__.__name__

    # Update the model results DataFrame
    model_results = update_model_results(model_name, train_rmse_formatted, val_rmse_formatted, model_results)

    return model_results


def poly_features(X_train, X_val, X_test):
    poly = PolynomialFeatures()
    X_train = poly.fit_transform(X_train)
    X_val = poly.transform(X_val)
    X_test = poly.transform(X_test)
    return X_train, X_val, X_test


def test_model(model_name, X_train, y_train, X_test, y_test, model_results=None):
    """
    Train a machine learning model and evaluate its performance on both training and validation sets.

    Parameters:
    model_name: A function that returns an instance of the machine learning model.
    X_train (DataFrame): Feature matrix for training data.
    y_train (Series): Target vector for training data.
    X_val (DataFrame): Feature matrix for validation data.
    y_val (Series): Target vector for validation data.
    model_results (DataFrame, optional): Existing DataFrame to store results.

    Returns:
    model: Trained machine learning model.
    model_results: Updated DataFrame containing model name and RMSE results.
    """
    # Fit the model on the training data
    model = model_name()
    model.fit(X_train, y_train)
    
    # # Make predictions on the training set
    # train_preds = model.predict(X_train)
    
    # # Calculate RMSE on the training set
    # train_rmse = eval_model(y_train, train_preds)
    
    # Make predictions on the validation set
    test_preds = model.predict(X_test)
    
    # Calculate RMSE on the validation set
    test_rmse = eval_model(y_test, test_preds)
    
    # Print RMSE values for training and validation sets (formatted)
    test_rmse_formatted = "${:,.2f}".format(test_rmse)
    #print(f'The train RMSE is {train_rmse_formatted}.')
    print(f'The test RMSE is {test_rmse_formatted}.')




# def train_model(model, X_train, y_train, X_val, y_val):
#     """
#     Train a machine learning model and evaluate its performance on both training and validation sets.
    
#     Parameters:
#     model: The machine learning model to be trained.
#     X_train (DataFrame): Feature matrix for training data.
#     y_train (Series): Target vector for training data.
#     X_val (DataFrame): Feature matrix for validation data.
#     y_val (Series): Target vector for validation data.
    
#     Returns:
#     model: Trained machine learning model.
#     """
#     # Fit the model on the training data
#     model.fit(X_train, y_train)
    
#     # Make predictions on the training set
#     train_preds = model.predict(X_train)
    
#     # Calculate RMSE on the training set
#     train_rmse = eval_model(y_train, train_preds)
    
#     # Make predictions on the validation set
#     val_preds = model.predict(X_val)
    
#     # Calculate RMSE on the validation set
#     val_rmse = eval_model(y_val, val_preds)
    
#     # Print RMSE values for training and validation sets
#     print(f'The train RMSE is {train_rmse}.')
#     print(f'The validate RMSE is {val_rmse}.')
    
#     return model