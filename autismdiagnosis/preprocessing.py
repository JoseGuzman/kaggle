"""
preprocessing.py

Upload competition data pandas
into Weights and biases

Author: Jose Guzman, sjm.guzman<at>gmail.com
Created: Wed May 25 19:40:48 CEST 2022

try python -m sklearnex preprocessing.py to run faster
 
"""
import numpy as np, pandas as pd

import os
import json 
from pathlib import Path

import wandb

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from mytransformers import ContinentTransformer
from mytransformers import WeightEncoder 
from mytransformers import DropperTransformer 


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

def create_weights( feature:str, train_dataset:pd.DataFrame, target_series:pd.Series) -> dict:
    """
    Generates a dictionary with the ratio of positive/negative outcomes
    """
    ncount = train_dataset[feature].nunique()

    # 1. Count and rank feature
    ranking = range(1, ncount+1)[::-1] # rank from max to 1
    # value_counts() returns the values as index
    myfeature = train_dataset[feature].value_counts().index.to_list()
    # dict of feature and number of ocurrences
    rank_feature = dict( zip(myfeature, ranking))
    assert len(rank_feature) == ncount, 'Number of unique features differs from dictionary size'

    # 2. Apply a pos/neg ratio to that rank
    for key, val in rank_feature.items():
        pos = np.sum( [ (target_series==1) & (train_dataset[feature] == key) ] )
        neg = np.sum( [ (target_series==0) & (train_dataset[feature] == key) ] )
        assert pos+neg == (train_dataset[feature] == key).sum() , 'positive + negative differ from features sizes'

        if neg == 0:
            ratio = 0 # avoid zero-division error

        else:
            ratio = pos/neg
        rank_feature[key] = ratio * val 

    # 3. normalize those ratios
    mymax = np.max( list( rank_feature.values() ) )
    mymin = np.min( list( rank_feature.values() ) )
    normalize = lambda x: (x-mymin)/(mymax-mymin)
    weight_feature = { key:normalize(val) for key, val in rank_feature.items() }

    return weight_feature
        

def remove_prefix(mystring:str)->str:
    """
    removes prefix until '--'
    e.g. 'z_scoring__age' to 'age'
    """
    # replace double underscore with *
    mystring = mystring.replace(r"__", "*")
    return mystring[ mystring.find("*")+1: ]

#=========================================================================
# login Weights and biases
#=========================================================================
#wandb_path = Path('~/.wandb/wandb.json').expanduser()
#with open(wandb_path) as fp:
#    mykey = json.load(fp)['key']
#wandb.login(key = mykey)

#=========================================================================
# Load austism-diagnosis pandas dataframe
#=========================================================================
with wandb.init(project="ASD", entity='neurohost', job_type="load-data") as run:
    pandas_dir = run.use_artifact('neurohost/ASD/dataframe:latest', type='dataset').download()
    train = pd.read_pickle( Path(pandas_dir,'train.pkl') )
    train_target = pd.read_pickle( Path(pandas_dir,'train_target.pkl') )
    test  = pd.read_pickle( Path(pandas_dir,'test.pkl') )

#=========================================================================
# Correct datasets  (join 'others' and '?')
#=========================================================================
# substitute 'others' by '?'
for df in (train,test):
    df.ethnicity[ df.ethnicity=='others'] = '?'
    df.relation[ df.relation=='Others' ] = '?'

#=========================================================================
# Add AgeGroup to datasets 
#=========================================================================
for df in (train,test):
    df.loc[(df.age < 14),  'AgeGroup'] = 'children'
    df.loc[(df.age >= 14) &  (df.age < 24),  'AgeGroup'] = 'youth'
    df.loc[(df.age >= 24) & (df.age < 64),  'AgeGroup'] = 'adult'
    df.loc[(df.age >= 64),  'AgeGroup'] = 'senior'

#=========================================================================
# Dictionary with region -> continent for ContinentTransformer
#=========================================================================
data = pd.read_csv('../input/country-mapping-iso-continent-region/continents2.csv')
continent = pd.Series(data.region.values, index=data.name).to_dict()
continent['Antarctica'] = 'Antarctica'

#=========================================================================
# Construct weights dictionaries to use in WeightEncoder 
#=========================================================================
mytraindataset = dict(train_dataset = train, target_series = train_target)
weight_country = create_weights(feature = 'contry_of_res', **mytraindataset ) 
weight_ethnicity = create_weights(feature = 'ethnicity', **mytraindataset ) 
weight_relation = create_weights(feature = 'relation', **mytraindataset ) 
weight_age = create_weights(feature = 'AgeGroup', **mytraindataset ) 

#=========================================================================
# Define Pipeline
#=========================================================================
country = ('country', ContinentTransformer(continent = continent)) # creates continent
w_country = ('weight_country', WeightEncoder(col_name = 'contry_of_res', weight = weight_country)) # creates w_contry_of_res 
w_ethnicity = ('weight_ethnicity', WeightEncoder(col_name = 'ethnicity', weight = weight_ethnicity)) # creates w_ethnicity
w_relation = ('weight_relation', WeightEncoder(col_name = 'relation', weight = weight_relation)) # creates w_ethnicity
w_age = ('weight_age', WeightEncoder(col_name = 'AgeGroup', weight = weight_age)) # creates w_AgeGroup

dropper = ('dropper', DropperTransformer(features = ['age_desc', 'gender', 'used_app_before', 'contry_of_res']))

z_scoring = ('z_scoring', StandardScaler(), [10, 14] ) # age, result
binarize = ('binarize', OneHotEncoder(sparse=False, drop= 'if_binary'), [12,13] ) # jaundice, austim
one_hot =  ('one_hot',  OneHotEncoder(sparse=False, handle_unknown='ignore'), [11,15,16,17] ) # ethnicity, relation, AgeGroup, continent

col_transformer = ColumnTransformer(transformers = (z_scoring, binarize, one_hot), remainder = 'passthrough')

col_preprocess = ('col_transformer', col_transformer)

preprocess = Pipeline( steps = (country, w_country, w_ethnicity, w_relation, w_age, dropper, col_preprocess))

# =====================================================================
# End Pipeline definition
# =====================================================================
foo = preprocess.fit(test)  # need to fit first to obtain  get_feature_names_out

mycols = [remove_prefix(mystring= col_name) for col_name in preprocess[-1].get_feature_names_out()]

# simply create a pandas now with the cols
pandarize = ('pandarized', FunctionTransformer(lambda x: pd.DataFrame(x, columns = mycols)))
preprocess = Pipeline(steps = (country, w_country, w_ethnicity, w_relation, w_age, dropper, col_preprocess, pandarize))


# =====================================================================
# Apply Pipeline 
# =====================================================================
# We first apply the preprocessing pipeline
Xtrain = preprocess.fit_transform(X = train)
Xtest = preprocess.fit_transform(X = test)

# check resulting variables are the same after preprocessing
assert (Xtrain.shape[1] == Xtest.shape[1]), 'train and set contain different number of independent variables'

# check the same number of independent variables
assert (Xtrain.shape[0] == train_target.shape[0]), 'train changed the number of observations'

#=========================================================================
# Log in Wandb DATASET->preprocess
#=========================================================================
with wandb.init(project='ASD', entity='neurohost', job_type='preprocessing') as run:

    # all steps together with their transformers
    info = {key:type(val).__name__ for key, val in preprocess.named_steps.items()}

    data_info = {
        'target':'Class/ASD', 
        'index_col':'ID', 
        'train_entries': Xtrain.shape[0], 
        'test_entries': Xtest.shape[0],
        'features' : Xtrain.shape[1] 
    }

    info.update(data_info)

    pre_data = wandb.Artifact(
        name = "preprocess", 
        type = "dataset",
        description = "Preprocessed data",
        metadata = info
    )

    mynames = ('train.csv', 'test.csv')
    for df, name  in zip((Xtrain, Xtest), mynames):
        df.to_csv(name, header=True)
        pre_data.add_file(name)
        #os.remove(name)

    train_target.to_csv('train_target.csv', header=True)
    pre_data.add_file('train_target.csv')
    #os.remove('train_target.csv')

    # and a table
    #pp_table = wandb.Table(data = Xtrain.values, columns = Xtrain.columns.to_list())
    #run.log({'mytable': pp_table})
    run.log_artifact(pre_data)
    