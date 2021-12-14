import math

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import datetime as dt
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


#https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html
def pls_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


    std = y_test.std()
    std = std["new_cases"]

    print(f"train std: {y_train.std()}")
    print(f"test std: {std}")
    for i in range(1, 40, 2):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")
        y_pred = pls.predict(X_test)

        mse = mean_squared_error(y_true = y_test, y_pred = y_pred, squared= False)
        mse = mse / std
        print(f"MSE: { mse}")
        #print(pls.score(X_test, y_test))

#http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html
def pls_cross_val(X, y):
    n = len(X)

    # 10-fold CV, with shuffle
    kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

    mse = []

    for i in np.arange(1, 40):
        pls = PLSRegression(n_components=i)
        # neg_mean_squared_error
        score = model_selection.cross_val_score(pls, X, y, cv=kf_10, scoring= 'r2'
                                                ).mean()
        # score = score * -1
        # score = math.sqrt(score)
        mse.append(score)

    print(mse)
    return mse


def pcr_cross_val(X, y):
    n = len(X)

    # 10-fold CV, with shuffle
    kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

    mse = []

    for i in np.arange(1, 40):
        pcr = make_pipeline(StandardScaler(), PCA(n_components=i), Ridge())

        score = model_selection.cross_val_score(pcr, X, y, cv=kf_10, scoring= 'r2'
                                                ).mean()
        # score = score * -1
        # score = math.sqrt(score)
        mse.append(score)


    print(mse)
    return mse

def l2(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    l2_lin_model = Ridge(fit_intercept=0)
    l2_lin_model.fit(X_train, y_train)
    l2_predict = l2_lin_model.predict(X_test)

    l2_r2 = mean_squared_error(y_pred=l2_predict, y_true=y_test, squared=False)
    print("Ridge score regression")
    print(l2_r2)

def pca(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for i in range(1, 40, 1):
        pcr = make_pipeline(StandardScaler(), PCA(n_components=i), Ridge())
        pcr.fit(X_train, y_train)
        pca = pcr.named_steps["pca"]
        pca_predict = pcr.predict(X_test)
        pca_r2 = r2_score(y_pred = pca_predict, y_true= y_test)
        print(f"PCA score. Components:{i}")
        print(pca_r2)



def main():
    data = pd.read_csv('owid-covid-data.csv')
    shift_var = -30

    uk_data = data.drop(data[data['iso_code'] != "GBR"].index)
    uk_data = uk_data.fillna(0)
    uk_data = uk_data.drop("iso_code", axis = 1)
    uk_data = uk_data.drop("continent", axis = 1)
    uk_data = uk_data.drop("location", axis = 1)

    averages = uk_data.mean()
    num_cases = uk_data[['new_cases']].copy()
    uk_data["new_cases"] = uk_data["new_cases"].shift(shift_var)

    #uk_data["new_cases"] = uk_data["new_cases"].fillna()

    uk_data['date'] = pd.to_datetime(uk_data['date'] )

    uk_data = uk_data.drop("tests_units", axis=1)

#Need to drop columns related to new cases.
    # uk_data['new_cases_smoothed'] = uk_data['new_cases_smoothed'].shift(shift_var)
    # uk_data['new_cases_per_million'] = uk_data["new_cases_per_million"].shift(shift_var)
    # uk_data['new_cases_smoothed_per_million'] = uk_data["new_cases_smoothed_per_million"].shift(shift_var)
    # uk_data['tests_per_case'] = uk_data["tests_per_case"].shift(shift_var)

#new dropped.
    uk_data['date'] = uk_data['date'].map(dt.datetime.toordinal)
    date_var = uk_data['date'].copy()
    uk_data = uk_data.shift(periods = shift_var)

    uk_data['date'] = date_var

    uk_data = uk_data.set_index(uk_data['date'])

    uk_data = uk_data.fillna(averages)
    #uk_data = uk_data.fillna(0)

    pls_score = pls_cross_val(uk_data, num_cases)
    pcr_score = pcr_cross_val(uk_data, num_cases)

    plt.plot(list(range(1, 40, 1)), pls_score, label = "PLS Score")
    plt.plot(list(range(1, 40, 1)), pcr_score, label = "PCR Score")

    # plt.xticks(list(range(0, 41, 2)))

    plt.xlabel("Number of components")
    plt.ylabel("Score (R^2)")
    plt.title("R-squared when predicting 30 days ahead")
    plt.legend()
    plt.show()

    plt.xlabel("Number of components")
    plt.ylabel("Root mean square error")
    plt.title("RMSE when predicting 30 days ahead")
    plt.legend()
    plt.show()






    # plt.xticks(list(range(0, 41, 2)))


    print(uk_data.columns)
    print(uk_data)


if __name__=="__main__":
    main()