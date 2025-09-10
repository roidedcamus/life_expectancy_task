import pandas as pd
import numpy as np

#  FUNCTION DEFINITIONS 
def get_nullcols(data):
    # Returns list of columns with any missing values
    return data.columns[data.isnull().any()].tolist()

def df_impute(data, cols):
    # Fills missing values in specified columns with their median
    for col in cols:
        if data[col].dtype != 'object':  # Only impute numeric columns
            median = data[col].median()
            data[col] = data[col].fillna(median)
    return data

def get_objcols(data):
    # Returns list of all categorical (object type) columns
    return data.select_dtypes(include=['object']).columns.tolist()

def df_ohe(data, obcols):
    # Performs one-hot encoding on categorical columns
    if obcols:  # Only encode if there are categorical columns
        data = pd.get_dummies(data, columns=obcols, drop_first=True)
    return data

#  MAIN PREPROCESSING FUNCTION
def preprocess_df(df, target='Life expectancy'):
    #  Clean column names: remove spaces, convert to lowercase
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+','_', regex=True)
    
    #  Update target name to match cleaned format
    target = target.strip().lower().replace(' ', '_')
    
    #  Drop rows where target variable is missing
    df = df.dropna(subset=[target])
    
    #  Drop columns with excessive missing values
    #  Also dropping year as it isn't meaningful
    df = df.drop(columns=['hepatitis_b', 'population', 'country', 'year'], errors='ignore')
    
    #  Log transforming a few columns
    for c in ['gdp', 'percentage_expenditure', 'measles', 'infant_deaths', 'under_five_deaths']:
        if c in df.columns:
            df[c] = np.log1p(df[c])


    #  Handle missing values in numeric columns
    nullcols = get_nullcols(df)
    df = df_impute(df, nullcols)
    
    #  One-hot encode all categorical variables
    objectcols = get_objcols(df)
    df = df_ohe(df, objectcols)
    
    #  Convert ALL data to numeric (force conversion)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    #  Drop any columns that are all NaN after conversion
    df = df.dropna(axis=1, how='all')
    
    #  Final check for any remaining missing values
    df = df.fillna(0)  # Fill any remaining NaN with 0
    
    #  Split into features (X) and target (y)
    X = df.drop(columns=[target])
    y = df[target]

    return X, y