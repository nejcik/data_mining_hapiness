import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# pd.set_option('display.expand_frame_repr', False)
# READ CSV (Pandas)
happy = pd.read_csv("_data/data2017.csv")
print(happy)

# Connect to regions: 

# def add_to_region(data):
    


# Classification - 3 things

# Clusterization - 3 things




# OUTLIERS
outliers=[]

def find_outlier(data, col):
    # IRQ funciton
    # med = median(data[col])
    df = pd.DataFrame({'outlier' : []})
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1 
    with pd.option_context('display.max_rows', -1, 'display.max_columns', 1):
        print(data[col] < (Q1 - 1.5 * IQR)) |(data[col] > (Q3 + 1.5 * IQR))





# CHANGE TO NUMPY ARRAY
# FutureWarning: Method .as_matrix will be removed in a future version. Use .values instead.
happy_np = happy.as_matrix()

# MAIN DATA happy_np[:,2:-1]
# Finding outliers: 
# Dopiero happy_np[5] ma zroznicowane dane.
# happy_np[0] - Kraj 
# happy_np[1] - Hapiness Rank
# happy_np[2] - Hapiness Score
# happy_np[3] - Whiskers High
# happy_np[4] - Whiskers Low

# happy.hist(figsize=(6,6))

find_outlier(happy, "Economy GDP")
print("*************************")
find_outlier(happy, "Family")
print("*************************")
find_outlier(happy, "Health Life Expectancy")
print("*************************")
find_outlier(happy, "Freedom")
print("*************************")
find_outlier(happy, "Generosity")
print("*************************")
find_outlier(happy, "Trust Government Corruption")
print("*************************")
# print(happy_np[:,5])