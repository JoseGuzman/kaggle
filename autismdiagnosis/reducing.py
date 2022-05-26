"""
reducing.py

This script is adapted from https://www.kaggle.com/code/wkirgsn/fail-safe-parallel-memory-reduction
Author: Kirgsn, 2018

How to use it in Kaggle:
1. Open a notebook and click File->Add Utility Script

Usage:
>>> from reducing import PandaReducer
>>> new_df = PandaReducer().reduce(df)
"""
import numpy as np
import pandas as pd

import time
import gc

from joblib import Parallel, delayed
from fastprogress import progress_bar

__all__ = ['PandaReducer']

def measure_time_mem(func):
    def wrapped_reduce(self, df:pd.DataFrame, *args, **kwargs)-> pd.DataFrame:
        
        # start time
        mem_init = df.memory_usage().sum() / self.memory_scale_factor
        start_time = time.time()
        
        # execute target function
        new_df = func(self, df, *args, **kwargs)
        
        # end time 
        mem_new = new_df.memory_usage().sum() / self.memory_scale_factor
        end_time = time.time()
        
        percentage = 100 * (mem_init - mem_new)/ mem_init 
        msg = 'Dataset reduced {0:.2f}% : {1:.2f} to {2:.2f} MB'.format(percentage, mem_init, mem_new)
        print(f'{msg} in {(end_time - start_time):.2f} seconds')
        
        gc.collect()
        return new_df
    
    return wrapped_reduce


class PandaReducer:
    """
    Class that takes a dict of increasingly big numpy datatypes to transform
    the data of a pandas dataframe into, in order to save memory usage.
    """
    memory_scale_factor = 1024**2  # memory in MB

    def __init__(self, use_categoricals:bool = True, n_jobs:int = -1):
        """
        :param use_categoricals: Whether the new pandas dtype "Categoricals"
                shall be used
        :param n_jobs: number of parallel jobs
        """
       
        #dict with np.dtypes-strings as keys
        self.conversion_table = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float32, ]
        }
        
        self.null_int = {   
            np.int8:  pd.Int8Dtype,
            np.int16: pd.Int16Dtype,
            np.int32: pd.Int32Dtype,
            np.int64: pd.Int64Dtype,
            np.uint8: pd.UInt8Dtype,
            np.uint16:pd.UInt16Dtype,
            np.uint32:pd.UInt32Dtype,
            np.uint64:pd.UInt64Dtype
        }
        
        self.use_categoricals = use_categoricals
        self.n_jobs = n_jobs

    def _type_candidates(self, k):
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    @measure_time_mem
    def reduce(self, df:pd.DataFrame, verbose:bool = False)-> pd.DataFrame:
        """
        Takes a dataframe and returns it with all data transformed to the
        smallest necessary types.

        :param df: pandas dataframe
        :param verbose: If True, outputs more information
        :return: pandas dataframe with reduced data types
        """
        ret_list = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(progress_bar(list(delayed(self._reduce)
                                                (df[c], c, verbose) for c in
                                                df.columns)))

        del df
        gc.collect()
        return pd.concat(ret_list, axis=1)

    def _reduce(self, s:pd.Series, colname:str, verbose:bool):
        """
        Real reduce function
        """
        try:
            isnull = False
            # skip NaNs
            if s.isnull().any():
                isnull = True
            # detect kind of type
            coltype = s.dtype
            if np.issubdtype(coltype, np.integer):
                conv_key = 'int' if s.min() < 0 else 'uint'
            elif np.issubdtype(coltype, np.floating):
                conv_key = 'float'
                asint = s.fillna(0).astype(np.int64)
                result = (s - asint)
                result = np.abs(result.sum())
                if result < 0.01:
                    conv_key = 'int' if s.min() < 0 else 'uint'
            else:
                if isinstance(coltype, object) and self.use_categoricals:
                    # check for all-strings series
                    if s.apply(lambda x: isinstance(x, str)).all():
                        if verbose: print(f'convert {colname} to categorical')
                        return s.astype('category')
                if verbose: print(f'{colname} is {coltype} - Skip..')
                return s
            # find right candidate
            for cand, cand_info in self._type_candidates(conv_key):
                if s.max() <= cand_info.max and s.min() >= cand_info.min:
                    if verbose: print(f'convert {colname} to {cand}')
                    if isnull:
                        return s.astype(self.null_int[cand]())
                    else:
                        return s.astype(cand)

            # reaching this code is bad. Probably there are inf, or other high numbs
            print(f"WARNING: {colname} doesn't fit the grid with \nmax: {s.max()} "
                f"and \nmin: {s.min()}")
            print('Dropping it..')
        except Exception as ex:
            print(f'Exception for {colname}: {ex}')
            return s