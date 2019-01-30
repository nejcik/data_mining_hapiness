import numpy as np
import pandas as pd

outliers=[]

def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

# READ CSV (Pandas)
happy = pd.read_csv("_data/2017.csv")
print happy

# CHANGE TO NUMPY ARRAY
happy_np = happy.as_matrix()

# outlier_datapoints = detect_outlier(happy)
# print(outlier_datapoints)
