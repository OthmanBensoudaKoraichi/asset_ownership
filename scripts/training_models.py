## Import libraries
import pandas as pd
import numpy as np
import timeit
import h5py
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

## Load dataset

labels = h5py.File('processed/labels.h5', 'r').get("labels")[...]
X= h5py.File('processed/data.h5', 'r').get("data")[...]

# labels1=labels[:]
# X1=X[:,:5000]
labels1=labels
X1=X

data=pd.DataFrame(labels1,columns=["wealth_index"])

## Sort values by asset ownership index
data = data.sort_values(by=["wealth_index"], ).reset_index(drop = False)

## Discretize the Y label

# Define how many categories we want
num_cat = 10

# Get the number of observations per category
num_obs = data.shape[0]/num_cat

# Create a Y discrete column with an arbitrary value of 1
data["Y_discrete"] = 1

# Create a column corresponding to the category number
for index, row in data.iterrows():
      data["Y_discrete"].iloc[index] = (index // num_obs) + 1


labels_discrete = data.sort_values(by=["index"],)["Y_discrete"]

## Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X1, labels_discrete , test_size = 0.2, random_state = 42)

# Convert them to numpy arrays
Y_train = Y_train.to_numpy(dtype=None, copy=False)
Y_test = Y_test.to_numpy(dtype=None, copy=False)

## Naive Bayes model

# Create a Naive Bayes instance
clf = GaussianNB()

# Fit the model on the training set
start2 = timeit.default_timer()
clf.fit(X_train, Y_train)
# Predict for the test set
pred = clf.predict(X_test)
print("naive bayes fit + prediction")
print(timeit.default_timer()-start2)

# Change back our results to a pandas dataframe
pred_pd = pd.DataFrame(pred,columns = ['prediction'])
actual_pd = pd.DataFrame(Y_test,columns = ['actual'])

# Merge our prediction and actual values
final_df =  pred_pd.merge(actual_pd, left_index = True, right_index = True)

# Create a column for the error

final_df["error"] = abs(final_df["prediction"] - final_df["actual"])/(num_cat-1)

accuracy=accuracy_score(Y_test, pred)*100
# Calculate the mean absolute error
print(f"Accuracy: {accuracy}%")
print(f"Distance ratio: {(1 - final_df['error'].mean())*100}%")

## Softmax regression
start2 = timeit.default_timer()
clf = LogisticRegression(random_state=42,max_iter =100, solver = 'lbfgs').fit(X_train, Y_train)
pred = clf.predict(X_test)
print("softmax fit + prediction")
print(timeit.default_timer()-start2)
#5k variables: 

# Change back our results to a pandas dataframe
pred_pd = pd.DataFrame(pred,columns = ['prediction'])
actual_pd = pd.DataFrame(Y_test,columns = ['actual'])

# Merge our prediction and actual values
final_df =  pred_pd.merge(actual_pd, left_index = True, right_index = True)

# Create a column for the error

final_df["error"] = abs(final_df["prediction"] - final_df["actual"])/(num_cat-1)
accuracy=accuracy_score(Y_test, pred)*100

# Calculate the mean absolute error
print(f"Accuracy: {accuracy}%")
print(f"Distance ratio: {(1 - final_df['error'].mean())*100}%")

## Preparation for continous models

## Train test split (with the same random state)
_ , _ , Y_train, Y_test = train_test_split(X1,labels1, test_size = 0.2, random_state = 42)

## Linear regression
start2 = timeit.default_timer()

reg = LinearRegression()

reg.fit(X_train, Y_train)

pred=reg.predict(X_test)

print("regression fit + prediction runtime: "+str(timeit.default_timer()-start2))
#5k variables: 20 seconds

rmse = mean_squared_error(Y_test, pred, squared=False)
r2= r2_score(Y_test, pred)

print(f"RMSE linear regression: {rmse}")
print(f"R2 linear regression: {r2}")

## LASSO
start2 = timeit.default_timer()

clf = linear_model.Lasso(alpha= 2)

clf.fit(X_train, Y_train)

pred = clf.predict(X_test)

print("LASSO regression fit + prediction runtime: "+str(timeit.default_timer()-start2))
#5k variables: 

rmse = mean_squared_error(Y_test, pred, squared=False)
r2= r2_score(Y_test, pred)

print(f"RMSE LASSO: {rmse}")
print(f"R2 LASSO: {r2}")
## RIDGE
start2 = timeit.default_timer()
clf = linear_model.Ridge(alpha=30)

clf.fit(X_train, Y_train)

pred = clf.predict(X_test)
print("RIDGE regression fit + prediction runtime: "+str(timeit.default_timer()-start2))


rmse = mean_squared_error(Y_test, pred, squared=False)
r2= r2_score(Y_test, pred)

print(f"RMSE Ridge: {rmse}")
print(f"R2 Ridge: {r2}")
