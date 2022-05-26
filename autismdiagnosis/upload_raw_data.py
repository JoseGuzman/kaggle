"""
upload_raw_data.py

Upload competition data into raw_data and pandas
into Weights and biases

Author: Jose Guzman, sjm.guzman<at>gmail.com
Created: Wed May 25 19:40:48 CEST 2022

 
"""
import os
import json 
from pathlib import Path

import pandas as pd 

import wandb
import kaggle

from typing import List, Tuple 
from reducing import PandaReducer

KG_DATASET = 'autismdiagnosis'

# if directory empty:
data_path = Path('../input/') / KG_DATASET # operator to extend path
if data_path.exists():
    print(f'{KG_DATASET} already exists in {data_path}')
    train_file = data_path / 'Autism_Prediction' / Path('train.csv')
    test_file = data_path  / 'Autism_Prediction' / Path('test.csv')
else:
    import zipfile, kaggle
    mykg_dataset = Path( KG_DATASET )
    # will read ~/.kaggle/kaggle.json
    kaggle.api.competition_download_cli( str(mykg_dataset) ) 
    myfilename = f'{mykg_dataset}.zip'
    with zipfile.ZipFile( myfilename ) as zf:
        zf.extractall(data_path)
        #zf.close()
    os.remove(myfilename)
    
#=========================================================================
# login Weights and biases
#=========================================================================
wandb_path = Path('~/.wandb/wandb.json').expanduser()
with open(wandb_path) as fp:
    mykey = json.load(fp)['key']
wandb.login(key = mykey)

#=========================================================================
# DATASET->raw_data
#=========================================================================
# 1. create a run 
run = wandb.init(project='ASD', entity='neurohost', job_type='load_data')

# 2. define name and type of artifact
info = {'source': 'https://www.kaggle.com/competitions/autismdiagnosis'}

raw_data = wandb.Artifact(
    name = 'raw_data',
    type = 'dataset',
    description = 'Based on Autism screening on adults dataset',
    metadata = info
    )

# 3. operate the artifact
raw_data.add_file( str(train_file) )
raw_data.add_file( str(test_file) )

# 3. Save and log artifact
run.log_artifact(raw_data)
run.finish()
    
#=========================================================================
# DATASET->pandas
#=========================================================================
run = wandb.init(project='ASD', entity='neurohost', job_type='load_data')

# Data Loading
def data_loader(file:Path, target:str=None, verbose:bool = False, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads csv file and return a tuple with a pandas dataset 
    containing all features, and a pandas Series with the 
    target variable.
    """
    
    data = pd.read_csv(file, **kwargs)
    df = PandaReducer().reduce(data) # see reduce.py in Utility Script 
    
    if target is not None:
        out = (df[target]).astype(int)
        df.drop([target], axis=1,  inplace=True) 
    else:
        out = target
    
    if verbose:
        print('The dataset contains {0} entries and {1} features'.format(*df.shape))
    
    return df, out


def dataframeit_and_log(train_file:Path, test_file:Path, info:dict):
    """
    creates dataframe versions from cvs files
    info :
        A dictionary with metadata to save
    """

    with wandb.init(project="ASD", entity='neurohost', job_type="load-data") as run:

        pd_data = wandb.Artifact(
            name = "dataframe", 
            type = "dataset",
            description = "Pandas DataFrame",
            metadata = info)
         
        
        train, train_target = data_loader(train_file, **info)
        test, _ = data_loader(file = test_file, target=None, verbose=True, index_col='ID')

        pd_data.new_file( train.to_pickle('train.pkl') )
        pd_data.new_file( train_target.to_pickle('train_target.pkl') )
        pd_data.new_file( test.to_pickle('test.pkl') )

        run.log_artifact(pd_data)

# declare which artifact we'll be using, if need be, download the artifact
#raw_data = run.use_artifact('raw_data:latest').download()
#train_file = Path(raw_data,'train.csv')
#test_file = Path(raw_data,'test.csv')

info = {'target':'Class/ASD', 'index_col':'ID'}
dataframeit_and_log(train_file, test_file, info)