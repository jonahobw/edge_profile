"""
Takes cleaned profile data and runs classifiers on it to predict model architecture.

Currently supports logistic regression and neural net.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import pandas as pd

from data_engineering import shared_data, get_data_and_labels, all_data, add_indicator_cols_to_input, remove_cols
from neural_network import Net
from config import SYSTEM_SIGNALS

def get_arch_pred_model(model_type, kwargs: dict = {}):
    arch_model = {"nn": NNArchPred}
    return arch_model[model_type](**kwargs)

class NNArchPred:

    def __init__(self, df=None, label = "model", verbose=True):
        self.verbose = verbose
        if df is None:
            df = all_data("zero_noexe_lots_models", no_system_data=True)
            exclude_cols = SYSTEM_SIGNALS
            exclude_cols.extend(["memcpy", "Malloc", "memset", "avg_us", "time_ms", "max_ms", "min_us"])
            df = remove_cols(df, substrs=exclude_cols)
        self.data = df
        self.label = label
        self.label_encoder = LabelEncoder()
        all_x, all_y = get_data_and_labels(self.data, shuffle=False, label=label)
        all_y_labeled = self.label_encoder.fit_transform(all_y)
        x_tr, x_test, y_train, y_test = train_test_split(all_x, all_y_labeled, random_state=42)
        self.x_tr = x_tr
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = len(all_y.unique())
        self.input_size = len(all_x.columns)
        print(f"Instantiating neural net with {self.num_classes} classes and input size of {self.input_size}")
        self.model = Net(input_size=self.input_size, num_classes=self.num_classes)
        self.x_tr = self.model.normalize(self.x_tr, fit=True)
        self.x_test = self.model.normalize(self.x_test)
        self.model.train_(self.x_tr, self.x_test, self.y_train, self.y_test, verbose=verbose)
        self.model.eval()
        self.validate_preds()
    
    def preprocessInput(self, x: pd.Series):
        x = add_indicator_cols_to_input(self.data, x, exclude=["file", "model", "model_family"])
        x = x.to_numpy(dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return x
    
    def predict(self, x: pd.Series, preprocess=True):
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.get_preds(x)
        #todo are they normalized to be confidence scores?
        pred = preds.argmax().cpu()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()
    
    def validate_preds(self):
        train_preds = self.model.get_preds(self.x_tr, normalize=False)
        pred = train_preds.argmax(dim=1).cpu()
        train_pred_labels = self.label_encoder.inverse_transform(np.array(pred))
        y_train_labels = self.label_encoder.inverse_transform(np.array(self.y_train))
        correct1 = sum(train_pred_labels == y_train_labels)
        print(f"X_train acc1: {correct1 / len(self.y_train)}")



class NNArchPredDebug:
    
    def __init__(self, df=None, label = "model"):
        if df is None:
            df = all_data("zero_noexe_lots_models").groupby("model").head(2)
        self.data = df
        self.label = label
        self.label_encoder = LabelEncoder()

        train = df.groupby("model").head(1)
        test = df.groupby("model").tail(1)

        x_tr, y_train = get_data_and_labels(train, shuffle=False, label=label)
        y_train = self.label_encoder.fit_transform(y_train)

        x_test, y_test = get_data_and_labels(test, shuffle=False, label=label)
        y_test = self.label_encoder.transform(y_test)

        self.x_tr = x_tr
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = len(self.data[label].unique())
        self.input_size = len(x_tr.columns)
        print(f"Instantiating neural net with {self.num_classes} classes and input size of {self.input_size}")
        self.model = Net(input_size=self.input_size, num_classes=self.num_classes)
        self.x_tr = self.model.normalize(self.x_tr, fit=True)
        self.x_test = self.model.normalize(self.x_test)
        self.model.train_(self.x_tr, self.x_test, self.y_train, self.y_test)
        self.model.eval()
        self.validate_preds()
    
    def preprocessInput(self, x: pd.Series):
        x = add_indicator_cols_to_input(self.data, x, exclude=["file", "model", "model_family"])
        x = x.to_numpy(dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return x
    
    def predict(self, x: pd.Series):
        x = self.preprocessInput(x)
        preds = self.model.get_preds(x)
        #todo are they normalized to be confidence scores?
        pred = preds.argmax().cpu()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()

    def validate_preds(self):
        train_preds = self.model.get_preds(self.x_tr, normalize=False)
        pred = train_preds.argmax(dim=1).cpu()
        train_pred_labels = self.label_encoder.inverse_transform(np.array(pred))
        y_train_labels = self.label_encoder.inverse_transform(np.array(self.y_train))
        correct1 = sum(train_pred_labels == y_train_labels)
        print(f"X_train acc1: {correct1 / len(self.y_train)}")


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
    net.train_(x_tr, x_test, y_train, y_test)


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
