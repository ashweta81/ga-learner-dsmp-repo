# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
print(df.columns[0:5])
print(df.info())

columns = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']

for col in columns:
    df[col] = df[col].replace({'\$':'', ',':''}, regex=True)

X = df.iloc[:, :-1]
y = df['CLAIM_FLAG']

count = y.value_counts()
print(count)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=6)


# Code ends here


# --------------
# Code starts here
for col in columns :
    X_train[col] = X_train[col].astype('float')

for col in columns:
    X_test[col] = X_test[col].astype('float')

X_train.isnull().sum()
X_test.isnull().sum()
# Code ends here


# --------------
# Code starts here
col1 = ['YOJ','OCCUPATION'] 
X_train.dropna(subset=col1, inplace=True)
X_test.dropna(subset=col1, inplace=True)

print(X_train.shape)
print(X_test.shape)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

col2 = ['AGE','CAR_AGE','INCOME','HOME_VAL']
for col in col2:
     X_train[col].fillna(X_train[col].mean(), inplace=True)
     X_test[col].fillna(X_test[col].mean(), inplace=True)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))


# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state=6)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print("The accuracy score is", score)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train,y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('The accuracy score is', score)
# Code ends here


