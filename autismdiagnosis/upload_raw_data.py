"""
upload_raw_data.py

Upload competition raw_data to Weights and biases
"""
import os
import json 
from pathlib import Path

import wandb
import kaggle

KG_DATASET = 'autismdiagnosis'

# login Weights and biases
wandb_path = Path('~/.wandb/wandb.json').expanduser()
with open(wandb_path) as fp:
    mykey = json.load(fp)['key']
wandb.login(key = mykey)

# if directory empty:
data_path = Path('../input/') / KG_DATASET # operator to extend path
if data_path.exists():
    print(f'{KG_DATASET} already exists in {data_path}')
else:
    import zipfile, kaggle
    mykg_dataset = Path( KG_DATASET )
    kaggle.api.competition_download_cli( str(mykg_dataset) ) # will read ~/.kaggle/kaggle.json
    myfilename = f'{mykg_dataset}.zip'
    with zipfile.ZipFile( myfilename ) as zf:
        zf.extractall(data_path)
        zf.close()
    os.remove(myfilename)
    

