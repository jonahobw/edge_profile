from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

from data_engineering import shared_data, get_data_and_labels


def logistic_reg(agg_csv_folder=None, system_data_only=False, no_system_data=False):
    if agg_csv_folder is None:
        agg_csv_folder = "zero_noexe"

    all_data = shared_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data)
    print(f"{len(list(all_data.columns))} Columns:\n{list(all_data.columns)}")
    all_x, all_y = get_data_and_labels(all_data, shuffle=False)
    x_tr, x_test, y_tr, y_test = train_test_split(all_x, all_y, random_state=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_tr, y_tr)
    acc = pipe.score(x_test, y_test)
    print(f"Logistic Regression acc: {acc}")
    # pred = pipe.predict(x_test)
    # for idx, pred in enumerate(pred):
    #     print(pred, y_test.iloc[idx])

def logistic_reg_rfe(agg_csv_folder=None):
    # todo implement with this article
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    if agg_csv_folder is None:
        agg_csv_folder = "zero_noexe"

    all_data = shared_data(agg_csv_folder)
    all_x, all_y = get_data_and_labels(all_data, shuffle=False)
    x_tr, x_test, y_tr, y_test = train_test_split(all_x, all_y)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_tr, y_tr)
    acc = pipe.score(x_test, y_test)
    print(f"Logistic Regression acc: {acc}")

if __name__ == '__main__':
    logistic_reg(system_data_only=True, no_system_data=False)