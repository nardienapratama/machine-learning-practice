import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = r'C:\Users\Nardiena A. Pratama\Documents\Personal Projects\machine-learning-practice\data\melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0) #dropna drops missing values
y = melbourne_data.Price # selecting the prediction target

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

print(X.describe())
print(X.head())
