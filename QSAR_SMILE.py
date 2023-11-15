#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:18:19 2023

@author: miko
"""

import pandas as pd
import numpy as np
from scipy.stats import randint
from scipy.stats import uniform
from rdkit import Chem
from rdkit.Chem import AllChem
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

'''
Loading data and conversion from SMILES to Morgan Fingerprints
'''

df_init = pd.read_csv('init_data.csv', delimiter = ';',decimal=',', on_bad_lines='skip')

#df_init = df_init.iloc[:,:6]

for index, row in df_init.iterrows():
    smiles = row['SMILES']
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2)
        bit_string = fp.ToBitString()
        df_init.loc[index, 'Morgan_fingerprints'] = bit_string
        
'''
Data preparation
'''     
  
X = df_init['Morgan_fingerprints'].apply(lambda x: list(map(int, list(x))))
X = pd.DataFrame(X.tolist())
y=pd.DataFrame(df_init['target variables'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Model initialization
'''

mlp = MLPRegressor(random_state=42)

'''
Find the best model parameters
'''
grid_param = {
    'hidden_layer_sizes': [(i,j) for i in range(10, 110, 10) for j in range(10, 110, 10)],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': uniform(0.0001, 0.05),
    'learning_rate': ['constant','adaptive', 'invscaling'],
    'warm_start': [True, False],
    'early_stopping': [True]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mlp_grid_search = GridSearchCV(mlp, grid_param, cv=cv, scoring='r2')
mlp_grid_search.fit(X_train, np.ravel(y_train))
best_model = mlp_grid_search.best_estimator_  

print("Grid search results:")
print(f"Best parameters: {mlp_grid_search.best_params_}")
print(f"Cross-validation results: {mlp_grid_search.best_score_}")
print(f'MSE: {mean_squared_error(y_test, mlp_grid_search.predict(X_test))}')
print(f'R^2: {r2_score(y_test, mlp_grid_search.predict(X_test))}')


random_param = {
    'hidden_layer_sizes': [(randint.rvs(1, 10), randint.rvs(1, 10),
                            randint.rvs(1, 10)) for _ in range(10)],
    'activation': ['logistic'],
    'solver': ['adam'],
    'alpha': uniform(0.0001, 0.001),
    'learning_rate': ['invscaling'],
    'warm_start': [True],
    'early_stopping': [True]
}
    
mlp_random_search = RandomizedSearchCV(mlp, random_param, n_iter=100, cv=cv, scoring='neg_root_mean_squared_error')
mlp_random_search.fit(X_train, np.ravel(y_train))
mlp_best_model = mlp_random_search.best_estimator_

print("Random search results:") 
print(f"Best parameters: {mlp_random_search.best_params_}")
print(f"Cross-validation results: {mlp_random_search.best_score_}")
print(f'MSE: {mean_squared_error(y_test, mlp_random_search.predict(X_test))}')
print(f'R^2: {r2_score(y_test, mlp_random_search.predict(X_test))}')

'''
Save the model for future use
'''

joblib.dump(best_model, 'Best regression model.pkl') 


