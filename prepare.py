import pandas as pd
from acquire import get_zillow_data

def rename_zillow(df):
    # Rename the columns
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area', 
                              'taxvaluedollarcnt':'value',
                              'yearbuilt':'year'})
    return df

def drop_zillow(df):
    # Drop nulls
    df = df.dropna()
    # Drop when values are 0 on either br or bath
    df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]
    # drop id column
    df.drop(columns=['parcelid'], inplace=True)
    df.drop(columns=['fips'], inplace=True)
    return df

def fips_map(df):
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

def datatype_zillow(df):
    # Convert data types
    df.bedrooms = df.bedrooms.astype('int')
    df.area = df.area.astype('int')
    df.value = df.value.astype('int')
    df.year = df.year.astype('int')
    #df.fips = df.fips.astype('int')
    return df

def zillow_pipeline():
    
    # run the acquire steps
    df = get_zillow_data()
    df = rename_zillow(df)
    df = fips_map(df)
    df = drop_zillow(df)
    df = datatype_zillow(df)
    
    #save changes to the csv file
    df.to_csv('zillow_data.csv',index=False)

    return df

def split_train_val_test(df):
    #split data
    seed = 42
     #first split 70/30 
    train, val_test = train_test_split(df, train_size=0.7,
                                    random_state=seed)
    #then plit the remainding 30 15/15
    val, test = train_test_split(val_test, train_size=0.5,
                                random_state=seed)
    return train, val, test

def scale_train_val_test(train, val, test):

    mms = MinMaxScaler()

    # Fit the scaler on the training data for all columns you want to scale
    columns_to_scale = ['year', 'area'] # 'taxamount',
    mms.fit(train[columns_to_scale])
    
    # Transform the specified columns for each dataset
    train[columns_to_scale] = mms.transform(train[columns_to_scale])
    val[columns_to_scale] = mms.transform(val[columns_to_scale])
    test[columns_to_scale] = mms.transform(test[columns_to_scale])
    
    return train, val, test
