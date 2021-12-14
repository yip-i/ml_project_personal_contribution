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

def pls_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    std = y_test.std()
    std = std["new_cases"]
    print(f"train max: {y_train.max()}")
    print(f"train min: {y_train.min()}")

    print(f"train std: {y_train.std()}")
    print(f"test std: {std}")
    for i in range(1, 40, 3):
        pls = PLSRegression(n_components=i)
        pls.fit(X_train, y_train)
        print(f"I = {i}")
        y_pred = pls.predict(X_test)

        print(f"MSE: {mean_squared_error(y_true = y_test, y_pred = y_pred, squared=False)}")
        #print(pls.score(X_test, y_test))


def main():
    data = pd.read_csv('owid-covid-data.csv')
    global_data = data[data['continent'].notna()]

    global_data = global_data.fillna(0)

    global_data = global_data.drop("iso_code", axis=1)
    global_data = global_data.drop("continent", axis=1)
    global_data = global_data.drop("location", axis=1)

    num_cases = global_data[['new_cases']].copy()

    global_data = global_data.drop("new_cases", axis=1)
    global_data = global_data.drop("tests_units", axis=1)

    global_data['date'] = pd.to_datetime(global_data['date'])
    global_data['date'] = global_data['date'].map(dt.datetime.toordinal)

    # Need to drop columns related to new cases.
    global_data = global_data.drop("new_cases_smoothed", axis=1)
    global_data = global_data.drop("new_cases_per_million", axis=1)
    global_data = global_data.drop("new_cases_smoothed_per_million", axis=1)
    global_data = global_data.drop("tests_per_case", axis=1)

    #Test dropping components
    global_data = global_data.drop("total_cases", axis=1)
    global_data = global_data.drop("total_cases_per_million", axis = 1)


    pls_regression(global_data, num_cases)
    print(global_data)
    print(global_data.columns)
if __name__=="__main__":
    main()