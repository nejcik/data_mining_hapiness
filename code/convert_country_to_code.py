import pandas as pd
import numpy as np
import pycountry
import argparse 
from dateutil.parser import parse

def valuation_formula(x):
	print(x)
	if x == "Bolivia":
		return 'BOL'	
	elif x=='Congo (Brazzaville)':
		return 'COG'
	elif x=='Congo (Kinshasa)': 
		return 'COD'
	elif x=='Czech Republic':
		return 'CZE'
	elif  x=='Iran' :
		return 'IRN'
	elif  x=='Ivory Coast': 
		return 'CIV'
	elif  x=='Kosovo' :
		return 'RKS'
	elif  x=='Macedonia': 
		return 'MKD'
	elif  x=='Moldova' :
		return 'MDA'
	elif  x=='North Cyprus': 
		return 'CYP'
	elif  x=='Palestinian Territories' :
		return 'PSE'	
	elif  x=='Russia':
		return 'RUS'
	elif  x=='South Korea' :
		return 'KOR'
	elif  x=='Syria' :
		return	'SYR'
	elif  x=='Tanzania':
		return 'TZA'
	elif  x=='Venezuela' :
		return 'VEN'
	elif  x=='Vietnam':
		return 'VNM'
	else:
		return (pycountry.countries.get(name=x).alpha_3)
	#Bolivia

#
#  Agument parser section
#
parser = argparse.ArgumentParser()
parser.add_argument("--data", dest = "datafile", default='D:/STUDIA/ed/NASZ/data2017.csv', help="Path to .csv file to use")
parser.add_argument("--path", dest = "source_path", default='./', help="Path to .csv file to save data.")
args = parser.parse_args()

# 'D:/STUDIA/ed/NASZ/_data/2017.csv'
# '../_data/2017.csv'
#
#  Logger section
#
datafile = args.datafile
source_path = args.source_path
df = pd.read_csv(datafile)

print(list(pycountry.countries))
print("********************")
print(df['Country'])

df=pd.DataFrame(data=df)
df['country_code'] = df.apply(lambda row: valuation_formula(row['Country']), axis=1)

df.head()

#print(df)

#df=pd.DataFrame(data=df)
name = 'data2017converted.csv'
path_to_converted = source_path + name
df.to_csv(path_to_converted)
