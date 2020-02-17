# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=len(df[df['fico']>700])/len(df)
p_b=len(df[df['purpose']=='debt_consolidation'])/len(df)

df1=df[df['purpose']=='debt_consolidation']

p_a_b=df1[df1['fico'].astype(float)>700].shape[0]/df1.shape[0]
print(p_a_b)

result=(p_a==p_a_b)
print(result)
# code ends here


# --------------
# code starts here
prob_lp=len(df[df['paid.back.loan']=='Yes'])/len(df)
prob_cs=len(df[df['credit.policy']=='Yes'])/len(df)
new_df=df[df['paid.back.loan']=='Yes']

prob_pd_cs=new_df[new_df['credit.policy']=='Yes'].shape[0]/new_df.shape[0]

bayes=prob_pd_cs*prob_lp/prob_cs
print(bayes)
# code ends here


# --------------
# code starts here
purpose=df['purpose'].value_counts()
plt.figure(figsize=(15,10))
purpose.plot(kind='bar')

df1=df[df['paid.back.loan']=='No']
plt.figure(figsize=(20,10))
y=df1['purpose'].value_counts()
y.plot(kind='bar')
# code ends here


# --------------
# code starts here
inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
plt.figure(figsize=(20,10))
plt.hist(df['installment'], bins=20)
plt.show()

plt.figure(figsize=(20,10))
plt.hist(df['log.annual.inc'], bins=20)
plt.show()
# code ends here


