import pandas as pd
import numpy as np
import argparse 
from dateutil.parser import parse
import random
import math as m
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from random import randint
import plotly.plotly as py
import pandas as pd
#plotly.tools.set_credentials_file(username='monifk', api_key='haslo123')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

#  df[(df['closing_price'] >= 99) & (df['closing_price'] <= 101)]

def classify_manual():
    group=[]
    for x in range(0,df['Country'].size):
        if ((df.iloc[x]['Happiness score'] >= 2.0) & (df.iloc[x]['Happiness score'] < 4.0)):
            group.append(1)
        elif ((df.iloc[x]['Happiness score'] >= 4.0 ) & (df.iloc[x]['Happiness score'] < 6.0)):
            group.append(2)
        elif ((df.iloc[x]['Happiness score'] >= 6.0)  & (df.iloc[x]['Happiness score'] <= 8.0)):
            group.append(3)

    df['Group'] = group

def pltcolor(lst):
    cols=[]
    for l in lst:
        if l==1.0:
            cols.append('red')
        elif l==2.0:
            cols.append('blue')
        elif l==3.0:
            cols.append('green')
    return cols

def classify_methods(X,y):
     X_train2, X_test2, y_train, y_test = train_test_split(X, y, test_size=0.73, random_state=42)
     print X_train2
     X_train = X_train2.drop(['Country code','Region', 'Happiness score',   'Country'], axis=1)
     X_test = X_test2.drop(['Country code','Region', 'Happiness score',   'Country'], axis=1)
     # print X_train
     # print y_train

     print("****************************************************************")

     scaler = StandardScaler()
     scaler.fit(X_train)
     X_train = scaler.transform(X_train)
     X_test = scaler.transform(X_test)
     # print('Standardized features\n')
     # print(str(X_train[:4]))

     lm = LinearRegression()
     lm.fit(X_train, y_train)
     y_pred = lm.predict(X_test)
     result_lm = pd.DataFrame({
     'Actual':y_test,
     'Predict':y_pred
     })
     result_lm['Diff'] = y_test - y_pred
     # print(result_lm.head())

     result_lm.to_csv('classify2018_1_gp.csv')

     clf = DecisionTreeClassifier().fit(X_train, y_train)
     print("\n")
     print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(clf.score(X_train, y_train)))
     print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))

     gnb = GaussianNB()
     gnb.fit(X_train, y_train)
     print("\n")
     print('Accuracy of GNB classifier on training set: {:.2f}'
          .format(gnb.score(X_train, y_train)))
     print('Accuracy of GNB classifier on test set: {:.2f}'
          .format(gnb.score(X_test, y_test)))

     knn = KNeighborsClassifier()
     knn.fit(X_train, y_train)
     print("\n")
     print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train, y_train)))
     print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test, y_test)))

     # pred = knn.predict(X_test)
     # print(confusion_matrix(y_test, pred))
     # print(classification_report(y_test, pred))

     logreg = LogisticRegression()
     logreg.fit(X_train, y_train)
     print("\n")
     print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(logreg.score(X_train, y_train)))
     print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(logreg.score(X_test, y_test)))

     print("****************************************************************")

def show_plot_lin_reg(X,y):
     X_train2, X_test2, y_train, y_test = train_test_split(X, y, test_size=0.73, random_state=42)
     # print X_train2
     # print y_train
     X_train = X_train2.drop(['Country code','Region','Happiness score',    'Country'], axis=1)
     X_test = X_test2.drop(['Country code','Region', 'Happiness score',   'Country'], axis=1)
     print("****************************************************************")

     scaler = StandardScaler()
     scaler.fit(X_train)
     X_train = scaler.transform(X_train)
     X_test = scaler.transform(X_test)
     # print('Standardized features\n')
     # print(str(X_train[:4]))

     lm = LinearRegression()
     lm.fit(X_train, y_train)
     y_pred = lm.predict(X_test)
     result_lm = pd.DataFrame({
     'Actual':y_test,
     'Predict':y_pred
     })
     # result_lm['Diff'] = y_test - y_pred
     print result_lm

     # rounded = int(result_lm["Predict"].round())
     items = [1, 2, 3]

     Region = X_test2['Region'].unique()
     print y 
     # Create the colors list using the function above
     cols=pltcolor(result_lm["Predict"].round())

     plt.scatter(x=result_lm["Predict"], y=X_test2['Region'], s = 20, c=cols)
     red_patch = mpatches.Patch(color='red', label='LOW')
     blue_patch = mpatches.Patch(color='blue', label='MID')
     green_patch = mpatches.Patch(color='green', label='HIGH')
     plt.title( "Klasyfikacja")
     legend = plt.legend(handles=[red_patch, green_patch, blue_patch], loc=4, fontsize='small', fancybox=True)
     plt.show()

     print("HALO")

#  MAP
     # y = [0]*y.ndim 
     # print y 

     # y[X[""]]

     # data = [ dict(
     # type = 'choropleth',
     # locations = X['Country code'],
     # z = y,
     # text = X_test2['Country'],
     # colorscale = [[0,"red"],[0.2,"blue"],[0.4,"rgb(120, 147, 247)"]],
     # autocolorscale = False,
     # reversescale = False,
     # marker = dict(
     #      line = dict (
     #           color = 'rgb(180,180,180)',
     #           width = 0.5
     #      ) ),
     # colorbar = dict(
     #      autotick = True,
     #      tickprefix = '',
     #      title = 'Happiness group [1-3]'),
     # ) ]

     # layout = dict(
     # title = 'World Happiness Groups',
     # geo = dict(
     # showframe = False,
     # showcoastlines = False,
     # projection = dict(
     #      type = 'Mercator'
     # )
     # )
     # )

     # fig = dict( data=data, layout=layout )
     # iplot( fig, validate=False, filename='d3-world-map-class' )
     # plot(fig, validate=False, filename='d3-world-map-class.html', image='png')
     

#
#  Agument parser section
#
parser = argparse.ArgumentParser()
parser.add_argument("--data", dest = "datafile",  default='D:/STUDIA/ed/NASZ/data2018.csv', help="Path to .csv file to use")
parser.add_argument("--path", dest = "dest_path", default='./', help="Path to .csv file to save data.")
args = parser.parse_args()

# Paths
# 'D:/STUDIA/ed/NASZ/data2017converted.csv'
# '~/Documents/SEM2/ED/data_mining_hapiness/_data/data2017converted.csv'

#
#  Logger section
#
datafile = args.datafile
dest_path = args.dest_path
df = pd.read_csv(datafile)
# Classify manually
classify_manual()

data_header = ['Country','Region','Happiness score','Whisker-high','Whisker-low','Dystopia (1.92) + residual',
'Explained by: GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy',
'Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption',
'Country code']

# all
X = df.drop(['Whisker-high','Whisker-low','Group'], axis=1)
y = df['Group']

classify_methods(X,y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 

# # plt.xlabel('height (cm)')
# # plt.ylabel('width (cm)')
# # plt.title("4-Class classification (k = %i, weights = '%s')"
# #            % (n_neighbors, weights))    
# # plt.show()
show_plot_lin_reg(X,y)