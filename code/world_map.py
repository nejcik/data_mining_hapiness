import plotly.plotly as py
import pandas as pd
#plotly.tools.set_credentials_file(username='monifk', api_key='haslo123')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import argparse 
from dateutil.parser import parse

#
#  Agument parser section
#
parser = argparse.ArgumentParser()
parser.add_argument("--data", dest = "datafile",  default='D:/STUDIA/ed/NASZ/data2017converted.csv', help="Path to .csv file to use")
args = parser.parse_args()

# Paths
# 'D:/STUDIA/ed/NASZ/data2017converted.csv'
# '~/Documents/SEM2/ED/data_mining_hapiness/_data/data2017converted.csv'

#
#  Logger section
#
datafile = args.datafile
df = pd.read_csv(datafile)

data = [ dict(
        type = 'choropleth',
        locations = df['country_code'],
        z = df['Happiness.Score'],
        text = df['Country'],
        colorscale = [[0,"rgb(220, 220, 220)"],[0.2,"rgb(150, 160, 180)"],[0.4,"rgb(120, 147, 247)"],\
            [0.6,"rgb(100, 130, 245)"],[0.7,"rgb(80, 110, 245)"],[1,"rgb(5, 10, 172)"]],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = True,
            tickprefix = '',
            title = 'Happiness score [0-10]'),
      ) ]

layout = dict(
    title = 'World Happiness Score',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
plot(fig, validate=False, filename='d3-world-map.html', image='png')