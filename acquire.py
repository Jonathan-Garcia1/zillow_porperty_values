import pandas as pd
from env import get_connection
import os

def acquire_zillow():
    """
    Acquire Zillow data from a SQL database.

    This function connects to a SQL database named 'zillow', executes a query to retrieve specific property data
    for the year 2017, and returns the data as a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing Zillow property data for the year 2017.

    Note:
        - This function relies on a helper function 'get_db_connection' to establish a database connection.
    """
    # Create a helper function to get the necessary database connection URL.
    def get_db_connection(database):
        return get_connection(database)

    # Connect to the SQL 'zillow' database.
    url = "zillow"

    # Use this query to get the data.
    sql_query = '''
                SELECT pred.parcelid, prop.bedroomcnt, prop.bathroomcnt, prop.calculatedfinishedsquarefeet, prop.taxvaluedollarcnt, prop.yearbuilt, prop.fips  
                FROM predictions_2017 AS pred
                LEFT JOIN properties_2017 AS prop ON pred.parcelid = prop.parcelid
                WHERE pred.transactiondate LIKE '2017%%'
                AND prop.propertylandusetypeid = 261;
                '''

    # Assign the data to a DataFrame.
    df = pd.read_sql(sql_query, get_connection(url))

    return df

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

def get_zillow_data():
    """
    Get Zillow data either from a CSV file or the SQL database.

    This function first checks if a CSV file named 'zillow.csv' exists. If it does, it reads the data from the CSV
    file into a DataFrame. If the CSV file doesn't exist, it calls the 'acquire_zillow' function to fetch the data
    from the SQL database, caches the data into a CSV file, and returns the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing Zillow property data.

    Note:
        - The data is cached in 'zillow.csv' to avoid repeated database queries.
    """
    if os.path.isfile('zillow.csv'):
        # If the CSV file exists, read in data from the CSV file.
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        # Read fresh data from the database into a DataFrame.
        df = acquire_zillow()
        # Cache the data by saving it to a CSV file.
        df.to_csv('zillow.csv')
    return df