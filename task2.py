import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

def train_test_split(  dataset: pd.DataFrame,
                       target_col: str, 
                       test_size: float,
                       stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    # TODO: Write the necessary code to split a dataframe into a Train and Test feature dataframe and a Train and Test 
    # target series 
    train_features = pd.DataFrame()
    test_features = pd.DataFrame()
    train_targets = pd.DataFrame()
    test_targets = pd.DataFrame()
    return train_features,test_features,train_targets,test_targets

class PreprocessDataset:
    def __init__(self, 
                 train_features:pd.DataFrame, 
                 test_features:pd.DataFrame,
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        # TODO: Add any state variables you may need to make your functions work
        return

    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded 
        one_hot_encoded_dataset = pd.DataFrame()

        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in 
        # the variable one_hot_encode_cols "one hot" encoded 
        one_hot_encoded_dataset = pd.DataFrame()

        return one_hot_encoded_dataset

    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column 
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in 
        # the variable min_max_scale_cols scaled to the min and max of each column 
        min_max_scaled_dataset = pd.DataFrame()
        return min_max_scaled_dataset
    
    def pca_train(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the train_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n 
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def pca_test(self) -> pd.DataFrame:
        # TODO: use PCA to reduce the test_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n 
        pca_dataset = pd.DataFrame()
        return pca_dataset

    def feature_engineering_train(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset

    def feature_engineering_test(self) -> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied 
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        feature_engineered_dataset = pd.DataFrame()
        return feature_engineered_dataset

    def preprocess(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        # TODO: Use the functions you wrote above to create train/test splits of the features and target with scaled and encoded values 
        # for the columns specified in the init function
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()
        return train_features,test_features