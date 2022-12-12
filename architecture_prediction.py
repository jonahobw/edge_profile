"""
Takes cleaned profile data and runs classifiers on it to predict model architecture.

Currently supports logistic regression and neural net.
"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import pandas as pd

from data_engineering import shared_data, get_data_and_labels, all_data, add_indicator_cols_to_input, remove_cols, softmax
from neural_network import Net
from config import SYSTEM_SIGNALS

def get_arch_pred_model(model_type, kwargs: dict = {}):
    if df is None:
        path = Path.cwd() / "profiles" / "quadro_rtx_8000"  / "zero_exe"
        df = all_data(path)
    arch_model = {"nn": NNArchPred}
    return arch_model[model_type](**kwargs)

class NNArchPred:

    def __init__(self, df, label = "model", verbose=True):
        self.verbose = verbose
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

        test_preds = self.model.get_preds(self.x_test, normalize=False)
        pred = test_preds.argmax(dim=1).cpu()
        test_pred_labels = self.label_encoder.inverse_transform(np.array(pred))
        y_test_labels = self.label_encoder.inverse_transform(np.array(self.y_test))
        correct1 = sum(test_pred_labels == y_test_labels)
        print(f"X_test acc1: {correct1 / len(self.y_test)}")


class LRArchPred:

    def __init__(self, df, label="model", verbose=True) -> None:
        self.verbose = verbose
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
        # self.pipe = make_pipeline(StandardScaler(), LogisticRegression())
        self.pipe = make_pipeline(StandardScaler(), Normalizer(), LogisticRegression())
        self.pipe.fit(self.x_tr, self.y_train)
        acc = self.pipe.score(self.x_test, self.y_test)
        print(f"Logistic Regression acc: {acc}")

    def preprocessInput(self, x: pd.Series):
        x = add_indicator_cols_to_input(self.data, x, exclude=["file", "model", "model_family"])
        x = x.to_numpy(dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return x

    def predict(self, x: pd.Series, preprocess=True):
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.pipe.decision_function(x)[0]
        preds = softmax(preds)
        #todo are they normalized to be confidence scores?
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()

class LRArchPredRFE:

    def __init__(self, df, label="model", verbose=True, rfe_num: int = 200) -> None:
        self.verbose = verbose
        self.data = df
        self.label = label
        self.label_encoder = LabelEncoder()
        all_x, all_y = get_data_and_labels(self.data, shuffle=False, label=label)
        all_y_labeled = self.label_encoder.fit_transform(all_y)
        x_tr, x_test, y_train, y_test = train_test_split(all_x, all_y_labeled, random_state=42)
        self.orig_cols = all_x.columns
        self.x_tr = x_tr
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = len(all_y.unique())
        self.input_size = len(all_x.columns)
        self.estimator = LogisticRegression()
        self.rfe_num = rfe_num
        self.rfe = RFE(estimator=self.estimator, n_features_to_select=self.rfe_num, verbose=10 if self.verbose else 0)
        self.pipe = make_pipeline(StandardScaler(), Normalizer(), self.rfe, self.estimator)
        self.pipe.fit(self.x_test, self.y_test)
        self.printFeatures()
        acc = self.pipe.score(self.x_test, self.y_test)
        print(f"Logistic Regression acc: {acc}")
    
    def printFeatures(self):
        print("Remaining Features:")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                print(f"Feature {i}:\t{col_name[:80]}")
    
    def printFeatureRank(self, save_path: Path = None, suppress_output: bool = False):
        if save_path is None and suppress_output:
            raise ValueError
        print("Feature Ranking (Note only ranks features which aren't part of the model):")
        if save_path is not None:
            save_file = Path(save_path) / "feature_ranking.txt"
            f = open(save_file, "w+")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                s = f"Rank 0:\t{col_name}"
                if save_path is not None:
                    f.write(s + '\n')
                if not suppress_output:
                    print(s[:80])
        ranking = {}
        ranks = self.rfe.ranking_
        for i, rank in enumerate(ranks):
            ranking[rank] = self.orig_cols[i]
        for i in range(len(ranking)):
            if i in ranking:
                s = f"Rank {i}:\t{ranking[i]}"
                if save_path is not None:
                    f.write(s + '\n')
                if not suppress_output:
                    print(s[:80])
        if save_path is not None:
            f.close()
        

    def preprocessInput(self, x: pd.Series):
        x = add_indicator_cols_to_input(self.data, x, exclude=["file", "model", "model_family"])
        x = x.to_numpy(dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return x

    def predict(self, x: pd.Series, preprocess=True):
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.pipe.decision_function(x)[0]
        preds = softmax(preds)
        #todo are they normalized to be confidence scores?
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()



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


def logistic_reg_rfe(agg_csv_folder=None):
    # todo implement with this article
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
    pass


if __name__ == '__main__':
    # neural_net("zero_noexe_lots_models", shared_data_only=False, indicator_only=True)
    # logistic_reg("zero_noexe_lots_models")
    # logistic_reg("zero_noexe_lots_models", system_data_only=True)
    # logistic_reg("zero_noexe_lots_models", no_system_data=True)
    # logistic_reg("zero_noexe_lots_models", shared_data_only=False, no_system_data=True)
    # logistic_reg("zero_noexe_lots_models", shared_data_only=False)
    exit(0)
