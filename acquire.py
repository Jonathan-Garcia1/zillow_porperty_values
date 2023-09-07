import pandas as pd
from env import get_connection
import os

def acquire_zillow():
    # create helper function to get the necessary connection url.
    def get_db_connection(database):
        return get_connection(database)

    # connect to sql zillow database
    url = "zillow"

    # use this query to get data    
    sql_query = '''
                SELECT pred.parcelid, prop.bedroomcnt, prop.bathroomcnt, prop.calculatedfinishedsquarefeet, prop.taxvaluedollarcnt, prop.yearbuilt, prop.fips  
                FROM predictions_2017 AS pred
                LEFT JOIN properties_2017 AS prop ON pred.parcelid = prop.parcelid
                WHERE pred.transactiondate LIKE '2017%%'
                AND prop.propertylandusetypeid = 261;
                '''

    # assign data to data frame
    df = pd.read_sql(sql_query, get_connection(url))
    
    return df

def get_zillow_data():
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        # Read fresh data from db into a dataframe
        df = acquire_zillow()
        # Cache data
        df.to_csv('zillow.csv')
    return df   
def rename_zillow(df):
    # Rename the columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area', 
                              'taxvaluedollarcnt':'value', 'yearbuilt':'year'})
    return df