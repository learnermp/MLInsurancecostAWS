import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('insurance.csv')

dataset['sex'] = dataset['sex'].map({'female': 0, 'male': 1})
dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})
dataset['region'] = dataset['region'].map({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4})

X = dataset.drop(['charges'], axis = 1)

y = dataset['charges']

# Create Column Transformer with 3 types of transformers
# num_features = X.select_dtypes(exclude="object").columns
# cat_features = X.select_dtypes(include="object").columns

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer

# oh_transformer = OneHotEncoder()

# preprocessor = ColumnTransformer(
#     [
#         ("OneHotEncoder", oh_transformer, cat_features)  
#     ]
# )



regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print("The cost of isurance will be $", model.predict([[19, 0, 27.9, 0, 1, 2]]))