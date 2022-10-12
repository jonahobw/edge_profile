"""
Takes cleaned profile data and runs classifiers on it to predict model architecture.

Currently supports logistic regression and neural net.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

from data_engineering import shared_data, get_data_and_labels, all_data
from neural_network import Net


def neural_net(agg_csv_folder=None, system_data_only=False, no_system_data=False, label=None, shared_data_only=True,
               indicator_only=False):
    """

    :param agg_csv_folder:
    :param system_data_only:
    :param no_system_data:
    :param label: either 'model' or 'model_family'
    :param shared_data_only: if true, use only features shared by all profiles, else use all features with indicator
        columns for incomplete features
    :return:
    """
    if not label:
        label = "model"
    if not agg_csv_folder:
        agg_csv_folder = "zero_noexe"

    if shared_data_only:
        data = shared_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data)
    else:
        data = all_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data, indicators_only=indicator_only)

    print(f"{len(list(data.columns))} Columns:\n{list(data.columns)}")
    all_x, all_y = get_data_and_labels(data, shuffle=False, label=label)
    label_encoder = LabelEncoder()
    all_y_labeled = label_encoder.fit_transform(all_y)
    x_tr, x_test, y_train, y_test = train_test_split(all_x, all_y_labeled, random_state=42)

    num_classes = len(all_y.unique())
    input_size = len(all_x.columns)

    print(f"Instantiating neural net with {num_classes} classes and input size of {input_size}")
    net = Net(input_size=input_size, num_classes=num_classes)
    x_tr = net.normalize(x_tr, fit=True)
    x_test = net.normalize(x_test)
    net.train(x_tr, x_test, y_train, y_test)


def logistic_reg(agg_csv_folder=None, system_data_only=False, no_system_data=False, label=None, shared_data_only=True,
                 indicator_only=False):
    """

    :param agg_csv_folder:
    :param system_data_only:
    :param no_system_data:
    :param label: either 'model' or 'model_family'
    :param shared_data_only: if true, use only features shared by all profiles, else use all features with indicator
        columns for incomplete features
    :return:
    """

    if not label:
        label = "model"
    if not agg_csv_folder:
        agg_csv_folder = "zero_noexe"

    if shared_data_only:
        data = shared_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data)
    else:
        data = all_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data, indicators_only=indicator_only)

    print(f"{len(list(data.columns))} Columns:\n{list(data.columns)}")

    all_x, all_y = get_data_and_labels(data, shuffle=False, label=label)
    x_tr, x_test, y_tr, y_test = train_test_split(all_x, all_y, random_state=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    pipe.fit(x_tr, y_tr)
    acc = pipe.score(x_test, y_test)
    print(f"Logistic Regression acc: {acc}")
    y_preds = pipe.predict(x_test)
    f1 = f1_score(y_test, y_preds, average='micro')
    print(f"Logistic Regression f1: {f1}")
    # pred = pipe.predict(x_test)
    # for idx, pred in enumerate(pred):
    #     print(pred, y_test.iloc[idx])

def logistic_reg_rfe(agg_csv_folder=None):
    # todo implement with this article
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    pass


if __name__ == '__main__':
    # neural_net("zero_noexe_lots_models", shared_data_only=False, indicator_only=True)
    logistic_reg("zero_noexe_lots_models")
    logistic_reg("zero_noexe_lots_models", system_data_only=True)
    logistic_reg("zero_noexe_lots_models", no_system_data=True)
    logistic_reg("zero_noexe_lots_models", shared_data_only=False, no_system_data=True)
    logistic_reg("zero_noexe_lots_models", shared_data_only=False)
    exit(0)