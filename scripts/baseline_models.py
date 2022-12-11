import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model
import timeit

#loading data
train_set=torch.load("data/pytorch/train_set.pt")
test_set=torch.load("data/pytorch/test_set.pt")
val_set=torch.load("data/pytorch/val_set.pt")

#dataframe to store results
baseline=pd.DataFrame(index=["rmse reg","r2 reg","rmse lasso","r2 lasso","rmse ridge","r2 ridge"],
                      columns=["train","val","test"])

#generating dataloaders
train_data = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
val_data = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
test_data = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

# ----- TRAIN -----
for X, y in train_data:
    X_train = X.numpy()
    y_train = y.numpy()

X_train = np.reshape(X_train, (len(X_train), 3*224*224))

## Linear regression
reg = LinearRegression()

reg.fit(X_train, y_train)

pred_train = reg.predict(X_train)

baseline.loc["rmse reg"]["train"]= mean_squared_error(y_train, pred_train, squared=False)
baseline.loc["r2 reg"]["train"] = r2_score(y_train, pred_train)


## LASSO
lasso = linear_model.Lasso(alpha=5)

lasso.fit(X_train, y_train)

pred_train = lasso.predict(X_train)

baseline.loc["rmse lasso"]["train"] = mean_squared_error(y_train, pred_train, squared=False)
baseline.loc["r2 lasso"]["train"] = r2_score(y_train, pred_train)

  

## RIDGE
ridge = linear_model.Ridge(alpha=5)

ridge.fit(X_train, y_train)

pred_train = ridge.predict(X_train)

baseline.loc["rmse ridge"]["train"] = mean_squared_error(y_train, pred_train, squared=False)
baseline.loc["r2 ridge"]["train"] = r2_score(y_train, pred_train)


# ----- VAL -----
for X, y in val_data:
    X_val = X.numpy()
    y_val = y.numpy()

X_val = np.reshape(X_val, (len(X_val), 3*224*224))

## Linear regression
pred_val = reg.predict(X_val)

baseline.loc["rmse reg"]["val"] = mean_squared_error(y_val, pred_val, squared=False)
baseline.loc["r2 reg"]["val"] = r2_score(y_val, pred_val)


## LASSO
pred_val = lasso.predict(X_val)

baseline.loc["rmse lasso"]["val"] = mean_squared_error(y_val, pred_val, squared=False)
baseline.loc["r2 lasso"]["val"] = r2_score(y_val, pred_val)


## RIDGE
pred_val = ridge.predict(X_val)

baseline.loc["rmse ridge"]["val"] = mean_squared_error(y_val, pred_val, squared=False)
baseline.loc["r2 ridge"]["val"] = r2_score(y_val, pred_val)


# ----- TEST -----
for X, y in test_data:
    X_test = X.numpy()
    y_test = y.numpy()

X_test = np.reshape(X_test, (len(X_test), 3*224*224))

## Linear regression
pred_test = reg.predict(X_test)

baseline.loc["rmse reg"]["test"]  = mean_squared_error(y_test, pred_test, squared=False)
baseline.loc["r2 reg"]["test"] = r2_score(y_test, pred_test)


## LASSO
pred_test = lasso.predict(X_test)

baseline.loc["rmse lasso"]["test"] = mean_squared_error(y_test, pred_test, squared=False)
baseline.loc["r2 lasso"]["test"] = r2_score(y_test, pred_test)


## RIDGE
pred_test = ridge.predict(X_test)

baseline.loc["rmse ridge"]["test"] = mean_squared_error(y_test, pred_test, squared=False)
baseline.loc["r2 ridge"]["test"] = r2_score(y_test, pred_test)


#storing pandas df as CSV
baseline.to_csv("data/results/results_baseline.csv",index=True)
