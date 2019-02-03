import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time                          # To time processes 
import warnings                      # To suppress warnings
import matplotlib.pyplot as plt      # For Graphics
import seaborn as sns
from sklearn import cluster, mixture # For clustering 
from sklearn.preprocessing import StandardScaler
import argparse 
from dateutil.parser import parse

 #.) K Means Clustering - K means clustering works by selecting centroids randomly and number of centroids are inputs to clustering algorithm.Once random centroids are selected then distance from each centroid for each observations are calculated and each observation data is allocated to a centroid to which distance of observation is minimum

#We will plots clusters of all the 6 dimensions with 2 dimensions in each 2 dimensional plot
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
#%matplotlib inline
warnings.filterwarnings('ignore')

# Method for K means clustering
def kmeans_Clustering(data,numberOfClusters):
       #Initializing Kmeans.cluster object was imported from sklearn in begining.
       kmeans = cluster.KMeans(n_clusters=numberOfClusters)
       # Fitting the input data and getting the cluster labels
       cluster_labels = kmeans.fit_predict(data)
       # Getting the cluster centers
       cluster_centers = kmeans.cluster_centers_
       cluster_centers.shape
       print(cluster_labels)
       return cluster_labels,cluster_centers
	   
#Plot the cluster
def plot_cluster(labels,centers,df_wh):
    #Getting number of columns
    numOfDimensions = df_wh.columns.size
    #Number of plots required for 6 dimension with 2 dimensions in each plot
    numberOfPlots = int(numOfDimensions/2)
    #Number of rows and columns for subplots
    fig,ax = plt.subplots(numberOfPlots,1, figsize=(10,10))
    for i,j in zip(range(0,numOfDimensions,2),range(0,numberOfPlots)):
         ax[j].scatter(df_wh.iloc[:, i], df_wh.iloc[:, i+1], c=labels, s=50, cmap='viridis')
         ax[j].scatter(centers[:,i], centers[:, i+1], c='black', s=200, alpha=0.5)
         #print(i)
    #plt.subplots_adjust(bottom=-0.5, top=1.5)
    print(numOfDimensions)
    plt.show()

	#axarr.plot(x,y)


#
#  Agument parser section
#
parser = argparse.ArgumentParser()
parser.add_argument("--data", dest = "datafile", default='D:/STUDIA/ed/NASZ/data2017.csv', help="Path to .csv file to use")
parser.add_argument("--path", dest = "source_path", default='./', help="Path to .csv file to save data.")
args = parser.parse_args()

# 'D:/STUDIA/ed/NASZ/data2017converted.csv'
# '../_data/data2017converted.csv'
#
#  Logger section
#
source_path = args.source_path
datafile = args.datafile
df_wh = pd.read_csv(datafile)

df_copy=df_wh
#df_wh=pd.DataFrame(data=df_wh)
df_wh.head()
df_wh=df_wh.rename(columns={'Economy..GDP.per.Capita.':'Economy_GDP_Per_Capita','Health..Life.Expectancy.':'Health_Life_Expectancy','Trust..Government.Corruption.':'Trust_Government_Corruption','Happiness.Rank':'Happiness_Rank','Happiness.Score':'Happiness_Score'})
df_wh=df_wh.filter(['Country','Economy_GDP_Per_Capita','Family','Health_Life_Expectancy','Freedom','Generosity','Trust_Government_Corruption'])
df_wh=df_wh.set_index('Country')


labels,centers = kmeans_Clustering(df_wh,6)   
plot_cluster(labels,centers,df_wh)

df_copy['clusters'] = labels
df_copy=pd.DataFrame(data=df_copy)
df_copy.head()

name = 'data2017clusters.csv'
path_df_clusters = source_path + name
df_copy.to_csv(path_df_clusters)

#plt.show()