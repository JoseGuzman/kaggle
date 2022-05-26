"""
preprocessing.py

Upload competition data into raw_data and pandas
into Weights and biases

Author: Jose Guzman, sjm.guzman<at>gmail.com
Created: Wed May 25 19:40:48 CEST 2022

 
"""
import json 
from pathlib import Path

import pandas as pd 

import wandb

from typing import List, Tuple 


#=========================================================================
# login Weights and biases
#=========================================================================
wandb_path = Path('~/.wandb/wandb.json').expanduser()
with open(wandb_path) as fp:
    mykey = json.load(fp)['key']
wandb.login(key = mykey)

#=========================================================================
# Load pandas dataframe
#=========================================================================
run = wandb.init()
pandas_dir = run.use_artifact('neurohost/ASD/dataframe:latest', type='dataset').download()
train = pd.read_pickle( Path(pandas_dir,'train.pkl') )
test  = pd.read_pickle( Path(pandas_dir,'test.pkl') )