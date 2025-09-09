import pandas as pd 

#  FUNCTION DEFINITIONS 

def get_nullcols(data):
    nc = []
    for i in data.columns:
        if data[i].isnull().any():
            nc.append(i)
    return nc

def df_impute(data, cols):
    for col in cols:
        median = data[col].median()
        data[col] = data[col].fillna(median)
    return data

def get_objcols(data):
    objcol = []
    for col in data.columns:
        if data[col].dtype == object:
            if col == 'Country':
                pass
            else:
                objcol.append(col)
    return objcol

def df_ohe(data, obcols):
    # does one hot encoding of the object columns
    # dropping one encoded column as it would be reduntant
    data = pd.get_dummies(data,columns=obcols, drop_first=True)
    # dropping the original object columns now
    data = data.drop(columns=obcols, axis = 1, errors='ignore')
    return data


#  MAIN 

def preprocess_df(df, target='life_expectancy'):

#  remove empty spaces etc from Col names
    df.columns = df.columns.str.strip().str.lower().str.replace(r'\s+','_', regex=True)
#  Missing Values
    #Firstly, we will remove the 5 rows for which 
    #the target variable - Life exptectancy is missing
    df = df.dropna(subset=[target])

    #We will also drop the Hep B and Population columns 
    #as they have a lot of missing values

    df = df.drop(columns=['hepatitis_b', 'population'])

    nullcols = get_nullcols(df)
    df = df_impute(df,nullcols)

    #  One-hot Encoding

    objectcols = get_objcols(df)
    df = df_ohe(df, objectcols)

    #  Splitting features and target

    feature_names = [c for c in df.columns if c!= target]
    X = df[feature_names]
    y = df[target]

    return X,y



