import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn import preprocessing

# Data import from folder
data_set = np.array(pd.read_csv("/Users/albertocenedese/Documents/Python/IML/Task_1b/train.csv"), dtype=np.float128)

# Data division into X abd Y
Y_set = data_set[:, 1]
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

#X = np.column_stack((X, np.ones(700)))  # constant part, w_0
print(X.shape)

# Normalization 
Y_mean = np.mean(Y_set)
Y_std = np.std(Y_set)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
print(np.tile(X_mean, (700,1)).shape)
print(np.tile(X_std, (700,1)).shape)

X_hat = (X - np.tile(X_mean, (700,1)))/np.tile(X_std, (700,1))
Y_hat = (Y_set - Y_mean)/Y_std


# Train - k fold validation and lasso regression:
N_fold = 10
kf = KFold(n_splits=N_fold, shuffle=False)
kf.get_n_splits(X_hat)

lambda_range = np.array([0.01, 0.1, 0.5, 1])

RMSEfield = np.zeros(lambda_range.shape)

# Optimization Loop: - crossvalidation
for lbda_ind in np.arange(lambda_range.shape[0]):
    lbda = lambda_range[lbda_ind]
    print(lbda)
    tempRMSE = np.array([])
    for train_index, test_index in kf.split(X_hat):
        X_train, X_test = X_hat[train_index], X_hat[test_index]
        Y_train, Y_test = Y_hat[train_index], Y_hat[test_index]
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
reg = Lasso(alpha=lambda_range[lb_opt], fit_intercept=False, tol=0.000001)
reg.fit(X_hat, Y_hat)
w = reg.coef_

w_1 = w * Y_std / X_std 
w_f = Y_mean - (np.dot(w_1.T, X_mean))*Y_std

print(pd.DataFrame(np.append(w_1, w_f)))
# Save
submSet = pd.DataFrame(np.append(w_1, w_f))
submSet.to_csv('/Users/albertocenedese/Documents/Python/IML/Task_1b/submission_w5.csv', header=False, index=False)

