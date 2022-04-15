from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE

from data_engineering import shared_data, get_data_and_labels
from neural_network import Net


def neural_net(agg_csv_folder=None, system_data_only=False, no_system_data=False, label=None):
    """

    :param agg_csv_folder:
    :param system_data_only:
    :param no_system_data:
    :param label: either 'model' or 'model_family'
    :return:
    """
    if not label:
        label = "model"
    if not agg_csv_folder:
        agg_csv_folder = "zero_noexe"

    all_data = shared_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data)
    print(f"{len(list(all_data.columns))} Columns:\n{list(all_data.columns)}")
    all_x, all_y = get_data_and_labels(all_data, shuffle=False, label=label)
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


def logistic_reg(agg_csv_folder=None, system_data_only=False, no_system_data=False, label=None):
    if not label:
        label = "model"
    if not agg_csv_folder:
        agg_csv_folder = "zero_noexe"

    all_data = shared_data(agg_csv_folder, system_data_only=system_data_only, no_system_data=no_system_data)
    print(f"{len(list(all_data.columns))} Columns:\n{list(all_data.columns)}")
    all_x, all_y = get_data_and_labels(all_data, shuffle=False, label=label)
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
    # logistic_reg(system_data_only=False, no_system_data=False)
    # logistic_reg("zero_noexe_lots_models")
    # logistic_reg("zero_noexe_lots_models", system_data_only=True)
    # logistic_reg("zero_noexe_lots_models", no_system_data=True)
    neural_net("zero_noexe_lots_models")
    exit(0)
