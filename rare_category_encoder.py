import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class RareCategoryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float=0.05):
        self.threshold = threshold
        
    def __rare_category_detector(self, X, y=None):
        X = pd.Series(X).copy()
        val_counts = X.value_counts(normalize=True)
        rare_cats = [*val_counts[val_counts < self.threshold].index]
        self.rare_cat_list.append(rare_cats)

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names = X.columns
        self.rare_cat_list = []
        X.apply(self.__rare_category_detector)
        return self
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x[x.isin(self.rare_cat_list[i])] = 'rare_category'
            X.iloc[:, i] = x
        return X
    
    def get_rare_cats(self):
        return self.rare_cat_list
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names

 
