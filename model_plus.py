import pandas as pd
import numpy as np
import os

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from explore_plus import features, hot_encode
from prepare import model_pipeline
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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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
    
    # Select columns that are either float or integer from train
    columns_to_scale = train.select_dtypes(include=['float', 'int']).columns.tolist()
    # Remove the target column
    columns_to_scale.remove('value')
    
    # Fit the scaler on the training data for specified columns
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

# def data_pipeline(df):
#     #df.drop(columns=['county','state'], inplace=True)
#     df = features(df)
#     df = df.drop(columns=['year', 'county', 'state', 'bath_bdrm_ratio', 'group_area', 'group_age']) 
#     # bedrooms	bathrooms area age year	county	state	total_rooms	age	bdrm_area_ratio	bath_bdrm_ratio	group_area	group_age
#     train, val, test = split_train_val_test(df)
#     train = train[(train['bedrooms'] <= 6)]
#     train = train[(train['bathrooms'] <= 6)]
#     train = train[(train['area'] >=699)]
#     train = train[(train['area'] <=5500)]
#     #train = train[(train['age'] <=114)]
#     train = train[(train['bdrm_area_ratio'] <=2000)] 
#     train = train[(train['bdrm_area_ratio'] >=100)] 
#     train, val, test = scale_train_val_test(train, val, test)
#     train = hot_encode(train)
#     val = hot_encode(train)
#     test = hot_encode(train)
#     X_train, y_train = xy_split(train)
#     X_val, y_val = xy_split(val)
#     X_test, y_test = xy_split(test)
#     return train, val, test, X_train, y_train, X_val, y_val, X_test, y_test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def mdata_pipeline(df):
    df = model_pipeline()
    #df.drop(columns=['county','state'], inplace=True)
    df = features(df)
    df = df.drop(columns=['year', 'county', 'state', 'bath_bdrm_ratio', 'group_area', 'group_age']) 
    # bedrooms	bathrooms area age year	county	state	total_rooms	age	bdrm_area_ratio	bath_bdrm_ratio	group_area	group_age
    df = df[(df['bedrooms'] <= 6)]
    df = df[(df['bathrooms'] <= 6)]
    df = df[(df['area'] >=699)]
    df = df[(df['area'] <=5500)]
    #df = df[(df['age'] <=114)]
    df = df[(df['bdrm_area_ratio'] <=2000)] 
    df = df[(df['bdrm_area_ratio'] >=100)] 
    train, val, test = split_train_val_test(df)
    train, val, test = scale_train_val_test(train, val, test)
    train = hot_encode(train)
    val = hot_encode(val)
    test = hot_encode(test)
    X_train, y_train = xy_split(train)
    X_val, y_val = xy_split(val)
    X_test, y_test = xy_split(test)
    return X_train, y_train, X_val, y_val, X_test, y_test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# def build_baselines(y_train):
#     baselines = pd.DataFrame({'y_actual': y_train,
#                           'y_mean': y_train.mean()})
#     return baselines

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def eval_baseline():
    """
    Evaluate the baseline model's performance using the root mean squared error (RMSE).

    Parameters:
        y_train (pd.Series): The target variable from the training dataset.

    Returns:
        float: The RMSE score representing the baseline model's performance.

    Note:
        - The function creates a baseline model by predicting the mean value of the training target variable for all samples.
        - It calculates the RMSE between the actual target values and the mean predictions.
        - The RMSE score quantifies the baseline model's performance.
    """
    df = model_pipeline()
    train, val, test = split_train_val_test(df)
    X_train, y_train = xy_split(train)
    
    baselines = pd.DataFrame({'y_actual': y_train, 'y_mean': y_train.mean()})
    
    return sqrt(mean_squared_error(baselines.y_actual, baselines.y_mean))

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def eval_model(y_actual, y_hat):
    """
    Evaluate a model's performance using the root mean squared error (RMSE).

    Parameters:
        y_actual (pd.Series): The actual target values.
        y_hat (pd.Series or np.array): The predicted target values.

    Returns:
        float: The RMSE score representing the model's performance.

    Note:
        - The function calculates the RMSE between the actual target values and the predicted values.
        - The RMSE score quantifies the model's performance, where lower values indicate better performance.
    """
    return sqrt(mean_squared_error(y_actual, y_hat))

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def update_model_results(model_name, train_rmse, val_rmse, model_results=None):
    """
    Update a DataFrame with model evaluation results (RMSE) for a given model.

    Parameters:
        model_name (str): The name or identifier of the model.
        train_rmse (float): The root mean squared error (RMSE) on the training dataset.
        val_rmse (float): The root mean squared error (RMSE) on the validation dataset.
        model_results (pd.DataFrame, optional): An existing DataFrame containing model results. Default is None.

    Returns:
        pd.DataFrame: An updated DataFrame with the new model's results.

    Note:
        - The function creates a DataFrame with the model's name and RMSE results on the training and validation datasets.
        - If `model_results` is provided, it concatenates the new results with the existing DataFrame.
        - If `model_results` is not provided, it creates a new DataFrame to store the results.
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

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def train_model(model_name, X_train, y_train, X_val, y_val, model_results=None):
    """
    Train a machine learning model, evaluate its performance, and update the model results DataFrame.

    Parameters:
        model_name (class): The machine learning model class (e.g., LinearRegression).
        X_train (pd.DataFrame): The feature matrix of the training dataset.
        y_train (pd.Series): The target variable of the training dataset.
        X_val (pd.DataFrame): The feature matrix of the validation dataset.
        y_val (pd.Series): The target variable of the validation dataset.
        model_results (pd.DataFrame, optional): An existing DataFrame containing model results. Default is None.

    Returns:
    model: Trained machine learning model.
    model_results: Updated DataFrame containing model name and RMSE results.

    Note:
        - The function trains a machine learning model on the provided training data.
        - It evaluates the model's performance on both the training and validation sets using RMSE.
        - RMSE values are printed for both sets in a formatted manner.
        - The model name is extracted from the class and used for updating the model results DataFrame.
        - If `model_results` is provided, it is updated with the new model's results.
        - If `model_results` is not provided, a new DataFrame is created to store the results.
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

    return model, model_results

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def poly_features(X_train, X_val, X_test):
    """
    Generate polynomial features for the given feature matrices.

    Parameters:
        X_train (pd.DataFrame): The feature matrix of the training dataset.
        X_val (pd.DataFrame): The feature matrix of the validation dataset.
        X_test (pd.DataFrame): The feature matrix of the test dataset.

    Returns:
        X_train (pd.DataFrame): The transformed feature matrix with polynomial features for the training dataset.
        X_val (pd.DataFrame): The transformed feature matrix with polynomial features for the validation dataset.
        X_test  (pd.DataFrame): The transformed feature matrix with polynomial features for the test dataset.

    Note:
        - The function generates polynomial features for the provided feature matrices using PolynomialFeatures.
        - It returns the transformed feature matrices for training, validation, and test datasets separately.
    """
    # Create a PolynomialFeatures object
    poly = PolynomialFeatures()
    
    # Transform the feature matrices using polynomial features
    X_train = poly.fit_transform(X_train)
    X_val = poly.transform(X_val)
    X_test = poly.transform(X_test)
    
    return X_train, X_val, X_test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def test_model(model, X_test, y_test, model_results=None):
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
    # model = model_name()
    # model.fit(X_train, y_train)
    
    
    # # Make predictions on the training set
    # train_preds = model.predict(X_train)
    
    # # Calculate RMSE on the training set
    # train_rmse = eval_model(y_train, train_preds)
    
    # Make predictions on the test set
    test_preds = model.predict(X_test)
    
    # Calculate RMSE on the test set
    test_rmse = eval_model(y_test, test_preds)
    
    # Print RMSE values for training and validation sets (formatted)
    test_rmse_formatted = "${:,.2f}".format(test_rmse)
    #print(f'The train RMSE is {train_rmse_formatted}.')
    print(f'The test RMSE is {test_rmse_formatted}.')
    
    # Extract the name of the model class without the module path
    model_name = model.__class__.__name__
    
    # Update the model results DataFrame
    model_results = update_model_results(model_name, test_rmse_formatted, "", model_results)
    
    return model_results