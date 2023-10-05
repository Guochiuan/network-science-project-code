import numpy as np
import pandas as pd
import sklearn.cluster
import yellowbrick.cluster

class KmeansClustering:
    def __init__(self, 
                 train_features:pd.DataFrame,
                 test_features:pd.DataFrame,
                 random_state: int
                ):
        # TODO: Add any state variables you may need to make your functions work
        pass

    def kmeans_train(self) -> list:
        # TODO: train a kmeans model using the training data, determine the optimal value of k (between 1 and 10) with n_init set to 10 and return a list of cluster ids 
        # corresponding to the cluster id of each row of the training data
        cluster_ids = list()
        return cluster_ids

    def kmeans_test(self) -> list:
        # TODO: return a list of cluster ids corresponding to the cluster id of each row of the test data
        cluster_ids = list()
        return cluster_ids

    def train_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the training dataset with a new feature called kmeans_cluster_id
        output_df = pd.DataFrame()
        return output_df

    def test_add_kmeans_cluster_id_feature(self) -> pd.DataFrame:
        # TODO: return the test dataset with a new feature called kmeans_cluster_id
        output_df = pd.DataFrame()
        return output_df