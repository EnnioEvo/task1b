import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

# Data import from folder
# !! use the relative path
data_set = np.array(pd.read_csv("../data/train.csv"), dtype=np.float64)
print(type(data_set))

# Data division into X abd Y
Y = data_set[:, 1]
X_set = data_set[:, 2:]

# Definition of the non linear features
non_lin_f = {"squared": lambda x: np.power(x, 2),
             "exp": lambda x: np.exp(x),
             "cos": lambda x: np.cos(x)
             }
# Definition of other parameters:
wgt_Number = data_set.shape[1] - 2

X = X_set
# Modification of the features:
for key in non_lin_f:
    X = np.column_stack((X, np.array(non_lin_f[key](X_set))))

X = np.column_stack((X, np.ones(700)))

# Train - k fold validation and lasso regression:
N_fold = 5
kf = KFold(n_splits=N_fold, shuffle=False)
kf.get_n_splits(X)
lb_num = 5
lambda_range = np.logspace(-2.0, 0.5, num=lb_num)
RMSEfield = np.zeros(lb_num)
print(X.shape)
# Optimization Loop:
for lbda_ind in np.arange(lb_num):
    lbda = lambda_range[lbda_ind]
    tempRMSE = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # Test
        reg = Lasso(alpha=lbda, fit_intercept=False, tol=0.0001)
        reg.fit(X_train, Y_train)
        w = reg.coef_
        # Test
        tempRMSE = np.append(tempRMSE, mean_squared_error(Y_test, reg.predict(X_test)) ** 0.5)
    RMSEfield[lbda_ind] = np.mean(tempRMSE)

# Optimal Value
lb_opt = np.argmin(RMSEfield)

# Final Train:
reg = Lasso(alpha=lambda_range[lb_opt], fit_intercept=False, tol=0.0001)

reg.fit(X, Y)
w = reg.coef_

# Save
submSet = pd.DataFrame(w)
submSet.to_csv('submission_w3.csv', header=False, index=False)
