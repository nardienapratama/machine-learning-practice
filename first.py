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

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

#Fit model
print(melbourne_model.fit(X, y))

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))