# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data=pd.read_csv(path)
data['Rating'].plot(kind='hist')

data=data[data['Rating']<=5]
data['Rating'].plot(kind='hist')

#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()

percent_null=(total_null/data.isnull().count())
missing_data=pd.concat([total_null,percent_null], axis=1, keys=['Total','Percent'])
print(missing_data)

data=data.dropna()
total_null_1=data.isnull().sum()
percent_null_1= (total_null/data.isnull().count())
missing_data_1=pd.concat([total_null_1,percent_null_1], axis=1, keys=['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here
ax=sns.catplot(x='Category',y='Rating', data=data, kind='box', height=10)
ax.set_xticklabels(rotation=90)
ax.fig.suptitle("Rating vs Category [BoxPlot]")
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].astype(int)

le=LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])
x2=sns.regplot(x='Installs',y='Rating', data=data)
plt.title('Rating vs Installs [RegPlot]')

#Code ends here



# --------------
#Code starts here
data['Price'].value_counts()
data['Price']=data['Price'].str.replace('$','')
data['Price']=data['Price'].astype(float)
sns.regplot(x='Price',y='Rating',data=data)
plt.title('Rating vs Price[RegPlot]')
#Code ends here


# --------------

#Code starts here

#Finding the length of unique genres
print( len(data['Genres'].unique()) , "genres")

#Splitting the column to include only the first genre of each app
data['Genres'] = data['Genres'].str.split(';').str[0]

#Grouping Genres and Rating
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

print(gr_mean.describe())

#Sorting the grouped dataframe by Rating
gr_mean=gr_mean.sort_values('Rating')

print(gr_mean.head(1))

print(gr_mean.tail(1))

#Code ends here



# --------------

#Code starts here
print(data['Last Updated'].head(2))

data['Last Updated']=pd.to_datetime(data['Last Updated'])
print(data['Last Updated'].head(2))

max_date=data['Last Updated'].max()

data['Last Updated Days']=(max_date - (data['Last Updated'])).dt.days

sns.regplot(x='Last Updated Days', y='Rating', data=data)
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here


