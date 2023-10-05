import numpy as np
import pandas as pd

def find_dataset_statistics(dataset:pd.DataFrame,target_col:str) -> tuple[int,int,int,int,float]:
    # TODO: Write the necessary code to generate the following dataset statistics given a dataframe
    # and a target column name. 

    #Total number of records
    n_records = 0
    #Total number of columns 
    n_columns = 0
    #Number of records where target is negative
    n_negative = 0
    #Number of records where where target is positive
    n_positive = 0 
    # Percentage of instances of positive target value
    perc_positive =  0

    return n_records,n_columns,n_negative,n_positive,perc_positive