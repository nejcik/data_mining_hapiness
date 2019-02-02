import pandas as pd
import numpy as np
import pycountry

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

df = pd.read_csv('D:/STUDIA/ed/NASZ/data2017.csv')


df=pd.DataFrame(data=df)
df['country_code'] = df.apply(lambda row: valuation_formula(row['Country']), axis=1)

df.head()

#print(df)

#df=pd.DataFrame(data=df)
df.to_csv('data2017converted.csv')
