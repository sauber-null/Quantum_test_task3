import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

# import data
data_train = pd.read_csv('internship_train.csv')
data_test = pd.read_csv('internship_hidden_test.csv')

# write data into arrays for further work
x_train = data_train.iloc[:, :-1].values
y_train = data_train.iloc[:, -1].values
x_test = data_test.values

# standardize data
min_max = preprocessing.MinMaxScaler()
x_train_scaled = min_max.fit_transform(x_train)
x_test_scaled = min_max.fit_transform(x_test)
x_train = pd.DataFrame(x_train_scaled)
x_test = pd.DataFrame(x_test_scaled)

# use polynomial ridge regression to make prediction
pipe = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=10**-7))
pipe.fit(x_train, y_train)
poly_pred = pipe.predict(x_train)
poly_pred_test = pipe.predict(x_test)

# write prediction to a file
final_pred = open("Prediction.csv", "x")
np.savetxt(final_pred, poly_pred_test)

final_pred.close()
