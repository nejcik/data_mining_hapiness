import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go, offline as offline
#from plotly import graphs_obj as go , offline as offline

#---------------------import data---------------------------------------
df = pd.read_csv('D:/STUDIA/ed/NASZ/data2017.csv')

print('We have',len(df['Country'].unique()),'countries. Thus, we decided to look at the trend by major regions')
print(df.columns)

df_region=df.groupby(['Region'])['Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.'].mean().sort_values(by="Happiness.Score", ascending=False)

#-----------save data grouped by region to csv file----------------------
df_region_df=pd.DataFrame(data=df_region)
df_region_df.to_csv('region_vs_hapiness.csv')

print(df_region_df)

#------------------draw table--------------------------------------------
fig, ax = plt.subplots()
# hide axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=df_region.values, colLabels=df_region.columns,  loc='center')
fig.tight_layout()
#plt.show()

#--------------plot 30 most ranked coutries(by region)-------------------
print('Top 30 country\'s region checking. Western Europe listed most! Southern asia least counts.:')

plt.figure(figsize=(10,6))
list = df.sort_values(by=['Happiness.Rank'],ascending=True)['Region'].head(30).value_counts()
list.plot(kind = 'bar', color = 'blue')
plt.title('Top 30 country\'s region checking')
#plt.show()

#--------------heat_map------------------------------------------
#created in world_map.py

#--------------happiness_vs_gdp----------------------------------
plt.figure(figsize=(12,8))
sns.regplot(x='Economy..GDP.per.Capita.',y='Happiness.Score' ,data=df)

plt.show()

