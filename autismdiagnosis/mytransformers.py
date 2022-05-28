"""
mytransformers.py

Author: Jose Guzman, sjm.guzman<at>gmail.com
Created: Thu May 26 10:52:02 CEST 2022

"""
import pandas as pd
from typing import List, Tuple 

from sklearn.base import BaseEstimator, TransformerMixin

class ContinentTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature transform a Country in one of the 
    five continents of the world, 'Asia' 'Europe' 
    'Africa' 'Oceania' 'Americas', or 'nan'.
    It creates a 'continent' feature.
    
    To use it:
    >>> myregion = RegionTransformer(continent = data)
    >>> df = myregion.fit_transform(X = train)
    """

    def __init__(self, continent:dict) -> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        continent:  (dict) of countries/continent pairs.
        """

        self.continent = continent
        self.df = None
    
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get column names (necessary for Pipelines)
        """
        
        if self.df is None:
            mycols = ['None']
        else:
            mycols =  self.df.columns.tolist()
            
        return mycols
        

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        
        df = X.copy()
        
        # correct 'AmericanSamoa and 'Viet Nam'
        typo1 = df[df['contry_of_res'] == 'AmericanSamoa'].index
        typo2 = df[df['contry_of_res'] == 'Viet Nam'].index
        
        if len(typo1):
            df['contry_of_res'] = df['contry_of_res'].cat.add_categories('American Samoa')
            for i in typo1:
                df.loc[i,'contry_of_res'] = 'American Samoa'
        
        if len(typo2):
            df['contry_of_res'] = df['contry_of_res'].cat.add_categories('Vietnam')
            for i in typo2:
                df.loc[i,'contry_of_res'] = 'Vietnam'
        
        
        df['continent'] = df['contry_of_res'].map(self.continent)
        self.df = df
        
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        
        df = X.copy()
        
        # correct 'AmericanSamoa and 'Viet Nam'
        typo1 = df[df['contry_of_res'] == 'AmericanSamoa'].index.tolist()
        typo2 = df[df['contry_of_res'] == 'Viet Nam'].index.tolist()
        
        if len(typo1):
            df['contry_of_res'] = df['contry_of_res'].cat.add_categories('American Samoa')
            for i in typo1:
                df.loc[i,'contry_of_res'] = 'American Samoa'
        
        if len(typo2):
            df['contry_of_res'] = df['contry_of_res'].cat.add_categories('Vietnam')
            for i in typo2:
                df.loc[i,'contry_of_res'] = 'Vietnam'
        
        # Add region to dataset
        df['continent'] = df['contry_of_res'].map(self.continent)
        return df

class WeightEncoder(BaseEstimator, TransformerMixin):
    """
    Custom feature for Label encoding based on weight 
    of categorical variables. The weight is based on the 
    relative proportion associated with the success of the
    dependent variable.
    
    To use it:
    >>> myrank = {'united states': 1, 'spain':10}
    >>> myfreq = WeightEncoder(col_name = ['country_of_res'], weight = myrank)
    >>> df = myfreq.fit_transform(X = train)
    
    It creates a new feature called 'w_col_name'.
    """

    def __init__(self, col_name:str, weight:dict)-> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        col_name:  the variable to remove
        rank: (dict) containg the variable and frequency to 
        be substitued (eg. myrank = {'united states': 1, 'spain':10}
        
        """

        self.col_name = col_name
        self.weight = weight
        self.df = None
   
       

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        df = X.copy()
                
        df['w_' + self.col_name] = df[self.col_name].map(self.weight).fillna(0)   
        self.df = df
        
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        df = X.copy()
        
        # 0 if not found in dictionary
        df['w_' + self.col_name] = df[self.col_name].map(self.weight).fillna(0)
    
        return df

class DropperTransformer(BaseEstimator, TransformerMixin):
    """
    Custom feature dropper to add to custom Pipelines.
    To use it:
    >>> mydropper = DropperTransformer(features = ['age'])
    >>> df = mydropper.fit_transform(X = train)
    """

    def __init__(self, features:List[str])-> None:
        """
        Remove the list of features from a pandas 
        Dataframe object.
        
        Parameter
        ---------
        features:  (list) of variables to remove
        """

        self.features = features
        self.df = None
    
    
    def get_feature_names_out(self)-> List[str]:
        """
        Get column names (necessary for Pipelines)
        """
        if self.df is None:
            mycols = ['None']
        else:
            mycols =  self.df.columns.tolist()
            
        return mycols
        

    def fit(self, X:pd.DataFrame, y = None):
        """
        Remove the column lists and update dataset
        """
        df = X.copy()
        self.df = df.drop(self.features, axis = 1)
        
        return self
   
    def transform(self, X:pd.DataFrame = None) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with removed features.
        
        Parameter
        ---------
        dataframe:  Pandas DataFrame object
        """
        df = X.copy()
        
        # Drop features
        self.df = df.drop(self.features, axis = 1)
        return self.df


