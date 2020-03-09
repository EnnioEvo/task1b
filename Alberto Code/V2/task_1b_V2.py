import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

# Data import from folder
# !! use the relative path
data_set = np.array(pd.read_csv("../../data/train.csv"), dtype=np.float64)
print(type(data_set))

# Data division into X abd Y
Y = data_set[:, 1]
X_set = data_set[:, 2:]

# Definition of the non linear features
non_lin_f = {"squared": lambda x: np.power(x, 2),
             "exp": lambda x: np.exp(x),
             "cos": lambda x: np.cos(x)
             }

# Modification of the features:
X = X_set  # First feature, linear
for key in non_lin_f:  # Loop over the features in order to modify the data
    X = np.column_stack((X, np.array(non_lin_f[key](X_set))))

X = np.column_stack((X, np.ones(700)))  # paconstant part, w_0

# Train - k fold validation and lasso regression:
N_fold = 10
kf = KFold(n_splits=N_fold, shuffle=False)
kf.get_n_splits(X)

lambda_range = np.array([0.01, 0.1, 1])

RMSEfield = np.zeros(lambda_range.shape)

# Optimization Loop: - crossvalidation
for lbda_ind in np.arange(lambda_range.shape[0]):
    lbda = lambda_range[lbda_ind]
    print(lbda)
    tempRMSE = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        # Test
        reg = Lasso(alpha=lbda, fit_intercept=False, tol=0.0001)
        reg.fit(X_train, Y_train)
        # Save the tempRMSE
        tempRMSE = np.append(tempRMSE, mean_squared_error(Y_test, reg.predict(X_test)) ** 0.5)
    RMSEfield[lbda_ind] = np.mean(tempRMSE)  # Mean of the RMSE for a given lambda
    print(reg.coef_)  # Print of the coef

# Optimal Value
lb_opt = np.argmin(RMSEfield)  # find the index of the optimal value
print("lamda opt: ", lambda_range[lb_opt])
print(RMSEfield)

# Final Train:
reg = Lasso(alpha=lambda_range[lb_opt], fit_intercept=False, tol=1e-6)
reg.fit(X, Y)
w = reg.coef_

# Save
submSet = pd.DataFrame(w)
submSet.to_csv('../submission_w4.csv', header=False, index=False)
