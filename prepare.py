import pandas as pd
from acquire import get_zillow_data
from sklearn.model_selection import train_test_split

def rename_zillow(df):
    """
    Rename selected columns in a DataFrame containing Zillow property data.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Zillow property data to be processed.

    Returns:
        pd.DataFrame: The DataFrame with selected columns renamed
    """
    # Rename the columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area', 
                              'taxvaluedollarcnt':'value',
                              'yearbuilt':'year'})
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def drop_zillow(df):
    """
    Process a DataFrame containing Zillow property data by performing the following operations:
    
    1. Drop rows with missing values (nulls).
    2. Drop rows where either 'bedrooms' or 'bathrooms' has a value of 0.
    3. Drop columns 'parcelid' and 'fips'.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Zillow property data to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame with rows and columns dropped as specified.
    """
    # Drop rows with missing values (nulls)
    df = df.dropna()
    
    # Drop rows where either 'bedrooms' or 'bathrooms' has a value of 0
    df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]
    
    # Drop columns 'parcelid' and 'fips'
    df.drop(columns=['parcelid', 'fips'], inplace=True)
    
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def fips_map(df):
    """
    Map FIPS codes to county and state names and add them as new columns to a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Zillow property data to be processed.

    Returns:
        pd.DataFrame: The DataFrame with two new columns 'county' and 'state' added based on the 'fips' column.
    """
    # Mapping of FIPS codes to county and state names
    fips_to_county_state = {
        6059: {'county': 'Orange', 'state': 'CA'},
        6111: {'county': 'Ventura', 'state': 'CA'},
        6037: {'county': 'Los Angeles', 'state': 'CA'},
    }

    # Add new columns 'county' and 'state' based on 'fips'
    df['county'] = df['fips'].map(lambda x: fips_to_county_state[x]['county'])
    df['state'] = df['fips'].map(lambda x: fips_to_county_state[x]['state'])

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def datatype_zillow(df):
    """
    Convert data types of specific columns in a DataFrame containing Zillow property data.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Zillow property data to be processed.

    Returns:
        pd.DataFrame: The DataFrame with selected columns converted to the 'int' data type.
    """
    # Convert data types
    df.bedrooms = df.bedrooms.astype('int')
    df.area = df.area.astype('int')
    df.value = df.value.astype('int')
    df.year = df.year.astype('int')
    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def zillow_pipeline():
    """
    Execute a data processing pipeline for Zillow property data, including data acquisition, transformation,
    and saving to a CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame containing Zillow property data.

    Note:
        - This function calls several other functions in the pipeline to acquire, transform, and save the data.
        - The processed data is saved to a CSV file named 'zillow_data.csv'.
    """
    # Run the data acquisition and transformation steps
    df = get_zillow_data()
    df = rename_zillow(df)
    df = fips_map(df)
    df = drop_zillow(df)
    df = datatype_zillow(df)
    train, val, test = split_train_val_test(df)
    
    # Save the processed data to a CSV file
    df.to_csv('zillow_data.csv', index=False)

    return train, val, test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def model_pipeline():
    """
    Execute a data processing pipeline for Zillow property data, including data acquisition, transformation,
    and saving to a CSV file.

    Returns:
        pd.DataFrame: The processed DataFrame containing Zillow property data.

    Note:
        - This function calls several other functions in the pipeline to acquire, transform, and save the data.
        - The processed data is saved to a CSV file named 'zillow_data.csv'.
    """
    # Run the data acquisition and transformation steps
    df = get_zillow_data()
    df = rename_zillow(df)
    df = fips_map(df)
    df = drop_zillow(df)
    df = datatype_zillow(df)
    #train, val, test = split_train_val_test(df)
    
    # Save the processed data to a CSV file
    df.to_csv('zillow_data.csv', index=False)

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------


def split_train_val_test(df):
    """
    Split a DataFrame containing Zillow property data into training, validation, and test sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing Zillow property data to be split.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Three DataFrames representing the training, validation, and test sets.

    Note:
        - The data splitting is performed with a fixed random seed for reproducibility.
        - The data is split as follows:
          - 70% of the data is used for training.
          - The remaining 30% is split into validation (15%) and test (15%) sets.
    """
    # Set a fixed random seed for reproducibility
    seed = 42
    
    # First split: 70% for training, 30% for validation and test combined
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    
    # Second split: Split the remaining 30% into validation (15%) and test (15%)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    return train, val, test

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def scale_train_val_test(train, val, test):
    """
    Scale specified columns in training, validation, and test sets using Min-Max scaling.

    Parameters:
        train (pd.DataFrame): The training set DataFrame.
        val (pd.DataFrame): The validation set DataFrame.
        test (pd.DataFrame): The test set DataFrame.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Scaled versions of the training, validation, and test sets.

    Note:
        - Min-Max scaling is applied to the columns 'year' and 'area' to bring their values within the range [0, 1].
        - The scaling is performed separately for each dataset (training, validation, test) using the same scaler.
    """
    mms = MinMaxScaler()

    # Fit the scaler on the training data for all columns you want to scale
    columns_to_scale = ['year', 'area']
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test
