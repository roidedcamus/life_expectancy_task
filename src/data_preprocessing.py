import pandas as pd 
import matplotlib as plt

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

#read in data into a pandas object
df = pd.read_csv('../data/Life Expectancy.csv')

#  PREPROCESSING 

#  remove empty spaces etc from Col names
#df.rename(columns={"Life expectancy ": "Life_expectancy", do rest})


#  Missing Values

#Firstly, we will remove the 5 rows for which 
#the target variable - Life exptectancy is missing

df = df.dropna(subset=['Life expectancy '])

#We will also drop the Hep B and Population columns 
#as they have a lot of missing values

df = df.drop(columns=['Hepatitis B', 'Population'])

#print(df['GDP'].describe())

nullcols = get_nullcols(df)
df = df_impute(df,nullcols)

#  One-hot Encoding

objectcols = get_objcols(df)
df = df_ohe(df, objectcols)






