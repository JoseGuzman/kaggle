"""
upload_raw_data.py

Upload competition data into wandb artifacts:
* raw_data 
* dataframe

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


def dataframeit_and_log(train_file:Path, test_file:Path):
    """
    creates dataframe versions from cvs files. It will go in the
    artifact panel, into DATASET->dataframe
    info :
        A dictionary with metadata to save
    """


#=========================================================================
# login Kaggle if dataset is empty
#=========================================================================
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
wandb_project = dict(project='ASD', entity='neurohost')
with wandb.init(**wandb_project, job_type='load_data') as run:

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
    
#=========================================================================
# DATASET->pandas
#=========================================================================
with wandb.init(project="ASD", entity='neurohost', job_type="load-data") as run:
    # load and reduce panda DataFrame
    train, train_target = data_loader(train_file, target='Class/ASD', index_col='ID')
    test, _ = data_loader(file = test_file, target=None, verbose=True, index_col='ID')

    info = {
        'target':'Class/ASD', 
        'index_col':'ID', 
        'train_entries': train.shape[0], 
        'test_entries': test.shape[0],
        'features' : train.shape[1] 
    }

    pd_data = wandb.Artifact(
        name = "dataframe", 
        type = "dataset",
        description = "Pandas DataFrame",
        metadata = info)
     
    
    train.to_pickle('train.pkl')
    train_target.to_pickle('train_target.pkl')
    test.to_pickle('test.pkl')

    pd_data.add_file( 'train.pkl' )
    pd_data.add_file('test.pkl')
    pd_data.add_file('train_target.pkl')
    os.remove('train.pkl')
    os.remove('train_target.pkl')
    os.remove('test.pkl')

    run.log_artifact(pd_data)