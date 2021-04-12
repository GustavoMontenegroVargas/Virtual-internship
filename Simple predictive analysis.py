#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


df = df_0.copy()
df['customer_id'].dropna(inplace = True)
Salary_df= df[df['txn_description'] == 'PAY/SALARY'] #This is the initial dataset that contains only the PAY/SALARY description
print(len(Salary_df['customer_id'].unique())) #All of our customers have salary transactions in our data set. 
Salary_df.reset_index(inplace = True)
Salary_df.drop('index',axis = 1, inplace = True)
date1= []
for i in range(len(Salary_df)):
    date1.append(Salary_df['date'][i][3:5])
Salary_df['month'] = date1
#----------------------------------------------------------------------------
print(len(Salary_df[Salary_df['month'] == '08'])) #We need to know if there are 100 unique customers_ids by month
print(len(Salary_df[Salary_df['month'] == '09']))
print(len(Salary_df[Salary_df['month'] == '10']))

print(len(Salary_df['customer_id'][Salary_df['month'] == '08'].unique())) #I want to know if all the customers have salary transactions each month
print(len(Salary_df['customer_id'][Salary_df['month'] == '09'].unique()))
print(len(Salary_df['customer_id'][Salary_df['month'] == '10'].unique()))
#-----------------------------------------------------------------------------
Qsalary = Salary_df[['customer_id','amount']].groupby('customer_id').sum()
#Qsalary.reset_index(inplace = True)
Qsalary.columns = ['QS']
Qsalary['AS'] = 4*Qsalary['QS'] #The annual salary column
Qsalary.head()


# In[3]:



#-----------------------------------------------------------------------------
df_age = Salary_df[['customer_id','age']].groupby('customer_id').mean()
#df_age.reset_index(inplace = True)
df_age.columns = ['age']
Qsalary = Qsalary.join(df_age)
#-----------------------------------------------------------------------------
print('There are', len(df['customer_id'][df['txn_description'] == 'POS'].unique()), 'unique customers of POS transactions')
print('There are', len(df['customer_id'][df['txn_description'] == 'SALES-POS'].unique()), 'unique customers of SALES-POS transactions')
print('There are', len(df['customer_id'][df['txn_description'] == 'PAYMENT'].unique()), 'unique customers of PAYMENT transactions')
print('There are', len(df['customer_id'][df['txn_description'] == 'INTER BANK'].unique()), 'unique customers of INTER BANK transactions')
print('There are', len(df['customer_id'][df['txn_description'] == 'PHONE BANK'].unique()), 'unique customers of PHONE BANK transactions')
#------------------------------------------------------------------------------
df_aws = df[['customer_id','amount']][df['txn_description']!= 'PAY/SALARY'].groupby('customer_id').sum()
#df_aws.reset_index(inplace = True)
df_aws.columns = ['Total amount - salary']
Qsalary = Qsalary.join(df_aws)
#--------------------------------------------------------------------------------
df_trans = df[['customer_id','Transaction values']][df['txn_description']!= 'PAY/SALARY'].groupby('customer_id').sum()
#df_trans.reset_index(inplace = True)
df_trans.columns = ['Total transactions - salary t.']
Qsalary = Qsalary.join(df_trans)
#------------------------------------------------------------------------------
df['gender'].replace({'M':1, 'F':0},inplace = True)
#-----------------------------------------------------------------------------
df_gender = df[['customer_id','gender']].groupby('customer_id').mean()
#df_gender.reset_index(inplace = True)
Qsalary = Qsalary.join(df_gender)
#----------------------------------------------------------------------------
df_payment = df[['customer_id','amount']][df['txn_description'] == 'PAYMENT'].groupby('customer_id').sum()
#df_payment.reset_index(inplace = True)
df_payment.columns = ['Total_amount_PT']
Qsalary = Qsalary.join(df_payment)
#---------------------------------------------------------------------------
df_pos = df[['customer_id','amount']][df['txn_description'] == 'POS'].groupby('customer_id').sum()
#df_pos.reset_index(inplace = True)
df_pos.columns = ['Total_amount_PosT']
Qsalary = Qsalary.join(df_pos)
#---------------------------------------------------------------------------
Qsalary1 = Qsalary.copy()
#---------------------------------------------------------------------------
df_Sp = df[['customer_id','amount']][df['txn_description'] == 'SALES-POS'].groupby('customer_id').sum()
df_Sp.columns = ['Total_amount_SpT']
Qsalary1 = Qsalary1.join(df_Sp)
#Qsalary1.set_index('customer_id', inplace = True)
#Qsalary1 = Qsalary1.join(df_Sp)
#Qsalary1.reset_index(inplace = True)
Qsalary1['Total_amount_SpT'].fillna(0, inplace = True)
#----------------------------------------------------------------------------
df_credit = df[['customer_id','Transaction values']][df['movement'] == 'credit'].groupby('customer_id').sum()
#df_credit.reset_index(inplace = True)
df_credit.columns = ['Total credit T']
Qsalary1 = Qsalary1.join(df_credit)
#-----------------------------------------------------------------------------
df_debit = df[['customer_id','Transaction values']][df['movement'] == 'debit'].groupby('customer_id').sum()
#df_debit.reset_index(inplace = True)
df_debit.columns = ['Total debit T']
Qsalary1 = Qsalary1.join(df_debit)
#-----------------------------------------------------------------------------
df_avws = df[['customer_id','amount']][df['txn_description']!= 'PAY/SALARY'].groupby('customer_id').mean()
#df_avws.reset_index(inplace = True)
df_avws.columns = ['average amount ws']
Qsalary1 = Qsalary1.join(df_avws)
#------------------------------------------------------------------------------
df_location = Salary_df[['customer_id','Latitude','Longitude']].groupby('customer_id').mean()
#df_location.reset_index(inplace = True)
Qsalary1 = Qsalary1.join(df_location)

#-----------------------------------------------------------------------------
df_balance = Salary_df[['customer_id','balance']].groupby('customer_id').sum()
#df_balance.reset_index(inplace = True)
df_balance.columns = ['Balance_sum_ST']
Qsalary1 = Qsalary1.join(df_balance)
#--------------------------------------------------------------------------------
df_balance = Salary_df[['customer_id','balance']].groupby('customer_id').mean()
#df_balance.reset_index(inplace = True)
df_balance.columns = ['Balance_avg_ST']
Qsalary1 = Qsalary1.join(df_balance)
#------------------------------------------------------------------------------
df_TT = df[['customer_id','Transaction values']].groupby('customer_id').sum()
#df_TT.reset_index(inplace = True)
df_TT.columns = ['Total transactions'] 
Qsalary1 = Qsalary1.join(df_TT)
#---------------------------------------------------------------------------
Qsalary1['A expenses'] = 4* Qsalary1['Total amount - salary']
#---------------------------------------------------------------------------
Qsalary1['A payment amount'] = 4* Qsalary1['Total_amount_PT']
#---------------------------------------------------------------------------
Qsalary1['A incomes - expenses'] = Qsalary1['AS'] - Qsalary1['A expenses']




# In[4]:


Qsalary1.head()


# In[5]:


len(Qsalary1)


# In[6]:


Qsalary1.corr()[1:2]


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, sharex=False, sharey=True)
ax1.plot(Qsalary1['age'],Qsalary1['AS'], 'o')
ax1.set_title('Correlation c:-0.03')
ax1.set_ylabel('Annual salary')
ax1.set_xlabel('age')
#----------------------------------
ax2.plot(Qsalary1['A expenses'],Qsalary1['AS'], 'o')
ax2.set_title('Correlation c: 0.37')
ax2.set_xlabel('Annual expenses')
#---------------------------------
ax3.plot(Qsalary1['Total transactions'],Qsalary1['AS'], 'o')
ax3.set_title('Correlation c: 0.099')
ax3.set_xlabel('Total transactions.')
#---------------------------------
ax4.plot(Qsalary1['A incomes - expenses'],Qsalary1['AS'], 'o', color = 'red')
ax4.set_title('Correlation c: 0.92')
ax4.set_xlabel('Annual incomes - Annual expenses')
ax4.set_ylabel('Annual salary')
#---------------------------------
ax5.plot(Qsalary1['A payment amount'],Qsalary1['AS'], 'o',color = 'green')
ax5.set_title('Correlation c: 0.63')
ax5.set_xlabel('Annual "payment" amount')
#---------------------------------
ax6.plot(Qsalary1['Total_amount_PosT'],Qsalary1['AS'], 'o')
ax6.set_title('Correlation c: -0.069')
ax6.set_xlabel('Total_amount_PosT')
#---------------------------------


# ## Linear regression model

# In[8]:


LR_df = Qsalary1[['A incomes - expenses','Total_amount_PT','AS']]
X = LR_df[['A incomes - expenses','Total_amount_PT']]
y = LR_df['AS']
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
Linear_model = LinearRegression().fit(X_train,y_train)
#print('linear model coeff (w): {}'
     #.format(Linear_model.coef_))
#print('linear model intercept (b): {:.3f}'
     #.format(Linear_model.intercept_))
print('Annual salary regression model')
print('Accuracy of linear regression model on training set: {:.3f}'
     .format(Linear_model.score(X_train, y_train)))
print('Accuracy of linear regression model on test set: {:.3f}'
     .format(Linear_model.score(X_test, y_test)))


# ## Regression with Decision Tree

# In[9]:


DT_df = Qsalary1[['A incomes - expenses','Total_amount_PT','AS']]
#binned = pd.cut(DT_df['AS'], bins = [0,23000,90800,405000,1000000],
                #labels = ['L','L-M','M-H','H'] )
#DT_df['AS_b'] = binned

X = DT_df[['A incomes - expenses','Total_amount_PT']]
y = DT_df['AS']

from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state = 0)

clf = DecisionTreeRegressor(max_depth = 4,
                            min_samples_leaf = 8,random_state 
                            = 0).fit(X_train, y_train)
print('Annual salary decision tree')
print('Accuracy of DT classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of DT classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[12]:


from sklearn import tree
plt.figure()
tree.plot_tree(clf)


# In[ ]:




