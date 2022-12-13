"""
Takes cleaned profile data and runs classifiers on it to predict model architecture.

Currently supports logistic regression and neural net.
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
from abc import ABC, abstractmethod

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
import pandas as pd

from data_engineering import (
    shared_data,
    get_data_and_labels,
    all_data,
    add_indicator_cols_to_input,
    remove_cols,
    softmax,
)
from neural_network import Net
from config import SYSTEM_SIGNALS


def get_arch_pred_model(model_type, df=None, label=None, kwargs: dict = {}):
    if df is None:
        path = Path.cwd() / "profiles" / "quadro_rtx_8000" / "zero_exe"
        df = all_data(path)
    arch_model = {"nn": NNArchPred, "lr": LRArchPred, "lr_rfe": LRArchPredRFE}
    return arch_model[model_type](df=df, label=label, **kwargs)


class ArchPredBase(ABC):
    def __init__(self, df, name: str, label=None, verbose=True) -> None:
        if label is None:
            label = "model"
        self.verbose = verbose
        self.data = df
        self.name = name
        self.label = label
        self.label_encoder = LabelEncoder()
        all_x, all_y = get_data_and_labels(self.data, shuffle=False, label=label)
        self.orig_cols = all_x.columns
        all_y_labeled = self.label_encoder.fit_transform(all_y)
        x_tr, x_test, y_train, y_test = train_test_split(
            all_x, all_y_labeled, random_state=42
        )
        self.x_tr = x_tr
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = len(all_y.unique())
        self.input_size = len(all_x.columns)

        # overwritten by subclasses
        self.model = None

    def preprocessInput(self, x: pd.Series):
        x = add_indicator_cols_to_input(
            self.data, x, exclude=["file", "model", "model_family"]
        )
        x = x.to_numpy(dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        return x

    @abstractmethod
    def getConfidenceScores(self, x: pd.Series, preprocess=True) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: pd.Series, preprocess=True) -> Tuple[str, float]:
        raise NotImplementedError

    def topK(self, x: pd.Series, k: int = 3, preprocess=True) -> List[str]:
        """A list of the topK classes predicted with most likely classes first."""
        conf_scores = self.getConfidenceScores(x, preprocess=preprocess)
        # adapted from
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        indices = np.argpartition(conf_scores, -k)[-4:]
        # sort by highest confidence first
        indices = indices[np.argsort(conf_scores[indices])][::-1]
        topk = conf_scores[indices]
        return self.label_encoder.inverse_transform(topk)

    def topKConf(
        self, x: pd.Series, k: int = 3, preprocess=True
    ) -> List[Tuple[str, float]]:
        """Same as self.topK but with confidence scores"""
        conf_scores = self.getConfidenceScores(x, preprocess=preprocess)
        # adapted from
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        indices = np.argpartition(conf_scores, -k)[-4:]
        result = []
        for idx in indices:
            result.append(
                (
                    self.label_encoder.inverse_transform(np.array([idx]))[0],
                    float(conf_scores[idx]),
                )
            )
        result = sorted(result, key=lambda x: x[1], reverse=True)
        return result


class NNArchPred(ArchPredBase):
    def __init__(
        self, df, label=None, verbose=True, hidden_layer_factor=None, num_layers=None, name="nn"
    ):
        super().__init__(df=df, name=name, label=label, verbose=verbose)
        print(
            f"Instantiating neural net with {self.num_classes} classes and input size of {self.input_size}"
        )
        self.model = Net(
            input_size=self.input_size,
            num_classes=self.num_classes,
            hidden_layer_factor=hidden_layer_factor,
            layers=num_layers,
        )
        self.x_tr = self.model.normalize(self.x_tr, fit=True)
        self.x_test = self.model.normalize(self.x_test)
        self.model.train_(
            self.x_tr, self.x_test, self.y_train, self.y_test, verbose=verbose
        )
        self.model.eval()
        self.validate_preds()

    def getConfidenceScores(self, x: pd.Series, preprocess=True) -> np.ndarray:
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.model.get_preds(x)
        return preds.cpu().numpy()

    def predict(self, x: pd.Series, preprocess=True) -> Tuple[str, float]:
        preds = self.getConfidenceScores(x, preprocess)
        pred = preds.argmax()
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


class LRArchPred(ArchPredBase):
    def __init__(self, df, label=None, verbose=True, name = "lr", multi_class: str="auto", penalty: str = "l2"):
        super().__init__(df=df, name=name, label=label, verbose=verbose)
        # self.pipe = make_pipeline(StandardScaler(), LogisticRegression())
        self.model = make_pipeline(StandardScaler(), Normalizer(), LogisticRegression(multi_class=multi_class, penalty=penalty))
        self.model.fit(self.x_tr, self.y_train)
        acc = self.model.score(self.x_test, self.y_test)
        print(f"Logistic Regression acc: {acc}")

    def getConfidenceScores(self, x: pd.Series, preprocess=True) -> np.ndarray:
        if preprocess:
            x = self.preprocessInput(x)
        return softmax(self.model.decision_function(x)[0])

    def predict(self, x: pd.Series, preprocess=True) -> Tuple[str, float]:
        preds = self.getConfidenceScores(x, preprocess)
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()


class LRArchPredRFE:
    def __init__(self, df, label=None, verbose=True, rfe_num: int = 200, name="lr_rfe", multi_class: str="auto", penalty: str = "l2") -> None:
        super().__init__(df=df, name=name, label=label, verbose=verbose)
        self.estimator = LogisticRegression(multi_class=multi_class, penalty=penalty)
        self.rfe_num = rfe_num
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.rfe_num,
            verbose=10 if self.verbose else 0,
        )
        self.model = make_pipeline(
            StandardScaler(), Normalizer(), self.rfe, self.estimator
        )
        self.model.fit(self.x_test, self.y_test)
        self.printFeatures()
        acc = self.model.score(self.x_test, self.y_test)
        print(f"Logistic Regression acc: {acc}")

    def getConfidenceScores(self, x: pd.Series, preprocess=True) -> np.ndarray:
        if preprocess:
            x = self.preprocessInput(x)
        preds = self.pipe.decision_function(x)[0]
        return softmax(preds)

    def predict(self, x: pd.Series, preprocess=True) -> Tuple[str, float]:
        preds = self.getConfidenceScores(x, preprocess)
        pred = preds.argmax()
        conf = preds[pred]
        label = self.label_encoder.inverse_transform(np.array([pred]))
        return label[0], conf.item()

    def printFeatures(self):
        print("Remaining Features:")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                print(f"Feature {i}:\t{col_name[:80]}")

    def printFeatureRank(self, save_path: Path = None, suppress_output: bool = False):
        if save_path is None and suppress_output:
            raise ValueError
        print(
            "Feature Ranking (Note only ranks features which aren't part of the model):"
        )
        if save_path is not None:
            save_file = Path(save_path) / "feature_ranking.txt"
            f = open(save_file, "w+")
        support = self.rfe.support_
        for i, col_name in enumerate(self.orig_cols):
            if support[i]:
                s = f"Rank 0:\t{col_name}"
                if save_path is not None:
                    f.write(s + "\n")
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
                    f.write(s + "\n")
                if not suppress_output:
                    print(s[:80])
        if save_path is not None:
            f.close()


if __name__ == "__main__":
    # neural_net("zero_noexe_lots_models", shared_data_only=False, indicator_only=True)
    # logistic_reg("zero_noexe_lots_models")
    # logistic_reg("zero_noexe_lots_models", system_data_only=True)
    # logistic_reg("zero_noexe_lots_models", no_system_data=True)
    # logistic_reg("zero_noexe_lots_models", shared_data_only=False, no_system_data=True)
    # logistic_reg("zero_noexe_lots_models", shared_data_only=False)
    exit(0)
