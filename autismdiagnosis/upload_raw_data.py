"""
upload_raw_data.py

Upload competition raw_data to Weights and biases

Author: Jose Guzman, sjm.guzman<at>gmail.com
Created: Wed May 25 19:40:48 CEST 2022

 
"""
import os
import json 
from pathlib import Path

import wandb
import kaggle

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
        zf.close()
    os.remove(myfilename)
    
#=========================================================================
# login Weights and biases
#=========================================================================
wandb_path = Path('~/.wandb/wandb.json').expanduser()
with open(wandb_path) as fp:
    mykey = json.load(fp)['key']
wandb.login(key = mykey)


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
    
