import numpy as np
import pandas as pd
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.feature_selection import RFE

class ModelMetrics:
    def __init__(self, model_type:str,train_metrics:dict,test_metrics:dict,feature_importance_df:pd.DataFrame):
        self.model_type = model_type
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.feat_imp_df = feature_importance_df
        self.feat_name_col = "Feature"
        self.imp_col = "Importance"
    
    def add_train_metric(self,metric_name:str,metric_val:float):
        self.train_metrics[metric_name] = metric_val

    def add_test_metric(self,metric_name:str,metric_val:float):
        self.test_metrics[metric_name] = metric_val

    def __str__(self): 
        output_str = f"MODEL TYPE: {self.model_type}\n"
        output_str += f"TRAINING METRICS:\n"
        for key in sorted(self.train_metrics.keys()):
            output_str += f"  - {key} : {self.train_metrics[key]:.4f}\n"
        output_str += f"TESTING METRICS:\n"
        for key in sorted(self.test_metrics.keys()):
            output_str += f"  - {key} : {self.test_metrics[key]:.4f}\n"
        if self.feat_imp_df is not None:
            output_str += f"FEATURE IMPORTANCES:\n"
            for i in self.feat_imp_df.index:
                output_str += f"  - {self.feat_imp_df[self.feat_name_col][i]} : {self.feat_imp_df[self.imp_col][i]:.4f}\n"
        return output_str


def calculate_naive_metrics(dataset:pd.DataFrame, target_col:str, naive_assumption:int) -> ModelMetrics:
    # TODO: Write the necessary code to calculate accuracy, recall, precision and fscore given a train and test dataframe
    # and a train and test target series and naive assumption 
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0
        }
    naive_metrics = ModelMetrics("Naive",train_metrics,test_metrics,None)
    return naive_metrics

def calculate_logistic_regression_metrics(dataset:pd.DataFrame, target_col:str, logreg_kwargs) -> tuple[ModelMetrics,LogisticRegression]:
    # TODO: Write the necessary code to train a logistic regression binary classifiaction model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test target series 
    # and keyword arguments for the logistic regrssion model
    model = LogisticRegression()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Use RFE to select the top 10 features 
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by ascending ranking then decending absolute value of Importance
    log_reg_importance = pd.DataFrame()
    log_reg_metrics = ModelMetrics("Logistic Regression",train_metrics,test_metrics,log_reg_importance)

    return log_reg_metrics,model

def calculate_random_forest_metrics(dataset:pd.DataFrame, target_col:str, rf_kwargs) -> tuple[ModelMetrics,RandomForestClassifier]:
    # TODO: Write the necessary code to train a random forest binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the random forest model
    model = RandomForestClassifier()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Reminder DONT use RFE for rf_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    rf_importance = pd.DataFrame()
    rf_metrics = ModelMetrics("Random Forest",train_metrics,test_metrics,rf_importance)

    return rf_metrics,model

def calculate_gradient_boosting_metrics(dataset:pd.DataFrame, target_col:str, gb_kwargs) -> tuple[ModelMetrics,GradientBoostingClassifier]:
    # TODO: Write the necessary code to train a gradient boosting binary classification model and calculate accuracy, recall, precision, fscore, 
    # false positive rate, false negative rate and area under the reciever operator curve given a train and test dataframe and train and test 
    # target series and keyword arguments for the gradient boosting model
    model = GradientBoostingClassifier()
    train_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    test_metrics = {
        "accuracy" : 0,
        "recall" : 0,
        "precision" : 0,
        "fscore" : 0,
        "fpr" : 0,
        "fnr" : 0,
        "roc_auc" : 0
        }
    # TODO: Reminder DONT use RFE for gb_importance
    # make sure the column of feature names is named Feature
    # and the column of importances is named Importance 
    # and the dataframe is sorted by decending absolute value of Importance
    gb_importance = pd.DataFrame()
    gb_metrics = ModelMetrics("Gradient Boosting",train_metrics,test_metrics,gb_importance)

    return gb_metrics,model