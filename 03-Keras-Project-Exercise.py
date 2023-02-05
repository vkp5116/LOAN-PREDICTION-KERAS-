#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.pieriandata.com"><img src="../Pierian_Data_Logo.PNG"></a>
# <strong><center>Copyright by Pierian Data Inc.</center></strong> 
# <strong><center>Created by Jose Marcial Portilla.</center></strong>

# # Keras API Project
# 
# ## The Data
# 
# I will be using a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club
# 
# 
# LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California.[3] It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform.
# 
# ### Goal of the project
# 
# Given historical data on loans given out with information on whether or not the borrower defaulted (charge-off), can I build a model thatcan predict wether or nor a borrower will pay back their loan? This way in the future when I get a new potential customer I can assess whether or not they are likely to pay back the loan. .
# 
# ### Data Overview

# ----
# -----
# There are many LendingClub data sets on Kaggle. Here is the information on this particular data set:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>LoanStatNew</th>
#       <th>Description</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>loan_amnt</td>
#       <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>term</td>
#       <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>int_rate</td>
#       <td>Interest Rate on the loan</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>installment</td>
#       <td>The monthly payment owed by the borrower if the loan originates.</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>grade</td>
#       <td>LC assigned loan grade</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>sub_grade</td>
#       <td>LC assigned loan subgrade</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>emp_title</td>
#       <td>The job title supplied by the Borrower when applying for the loan.*</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>emp_length</td>
#       <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>home_ownership</td>
#       <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>annual_inc</td>
#       <td>The self-reported annual income provided by the borrower during registration.</td>
#     </tr>
#     <tr>
#       <th>10</th>
#       <td>verification_status</td>
#       <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
#     </tr>
#     <tr>
#       <th>11</th>
#       <td>issue_d</td>
#       <td>The month which the loan was funded</td>
#     </tr>
#     <tr>
#       <th>12</th>
#       <td>loan_status</td>
#       <td>Current status of the loan</td>
#     </tr>
#     <tr>
#       <th>13</th>
#       <td>purpose</td>
#       <td>A category provided by the borrower for the loan request.</td>
#     </tr>
#     <tr>
#       <th>14</th>
#       <td>title</td>
#       <td>The loan title provided by the borrower</td>
#     </tr>
#     <tr>
#       <th>15</th>
#       <td>zip_code</td>
#       <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
#     </tr>
#     <tr>
#       <th>16</th>
#       <td>addr_state</td>
#       <td>The state provided by the borrower in the loan application</td>
#     </tr>
#     <tr>
#       <th>17</th>
#       <td>dti</td>
#       <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
#     </tr>
#     <tr>
#       <th>18</th>
#       <td>earliest_cr_line</td>
#       <td>The month the borrower's earliest reported credit line was opened</td>
#     </tr>
#     <tr>
#       <th>19</th>
#       <td>open_acc</td>
#       <td>The number of open credit lines in the borrower's credit file.</td>
#     </tr>
#     <tr>
#       <th>20</th>
#       <td>pub_rec</td>
#       <td>Number of derogatory public records</td>
#     </tr>
#     <tr>
#       <th>21</th>
#       <td>revol_bal</td>
#       <td>Total credit revolving balance</td>
#     </tr>
#     <tr>
#       <th>22</th>
#       <td>revol_util</td>
#       <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
#     </tr>
#     <tr>
#       <th>23</th>
#       <td>total_acc</td>
#       <td>The total number of credit lines currently in the borrower's credit file</td>
#     </tr>
#     <tr>
#       <th>24</th>
#       <td>initial_list_status</td>
#       <td>The initial listing status of the loan. Possible values are – W, F</td>
#     </tr>
#     <tr>
#       <th>25</th>
#       <td>application_type</td>
#       <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
#     </tr>
#     <tr>
#       <th>26</th>
#       <td>mort_acc</td>
#       <td>Number of mortgage accounts.</td>
#     </tr>
#     <tr>
#       <th>27</th>
#       <td>pub_rec_bankruptcies</td>
#       <td>Number of public record bankruptcies</td>
#     </tr>
#   </tbody>
# </table>
# 
# ---
# ----

# ## Imports and Data Extraction

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')


# In[15]:


print(data_info.loc['revol_util']['Description'])


# In[3]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[17]:


feat_info('mort_acc')


# ## Loading the data and other imports

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[106]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[20]:


df.info()


# 
# # Exploratory Data Analysis
# 
# **OVERALL GOAL: Get an understanding for which variables are important, view summary statistics, and visualize the data**

# **creating a countplot as shown below.**

# In[22]:


sns.countplot(data = df,x = "loan_status")


# **Creating a histogram of the loan_amnt column.**

# In[67]:


sns.set_style("whitegrid")
plt.figure(figsize = (12,4))
plt.hist(data = df, x = "loan_amnt",bins = 30)
plt.xlabel("loan_amnt")


# **Analyze the correlation between the continuous feature variables.**

# In[39]:


df.corr()


# **Visualizing this using a heatmap.**

# In[52]:


plt.figure(figsize = (12,9))
sns.heatmap(df.corr(),cmap = "coolwarm",annot = True)


# **Notice almost perfect correlation with the "installment" feature.I will analyze this feature further.**

# In[55]:


print(data_info.loc['installment']['Description'])


# In[56]:


print(data_info.loc['loan_amnt']['Description'])


# In[66]:


plt.scatter(data = df, x = "installment", y = "loan_amnt", marker="o",edgecolors = "yellow")
plt.xlabel("installments")
plt.ylabel("loan_amnt")


# **Creating a boxplot showing the relationship between the loan_status and the Loan Amount.**

# In[65]:


sns.boxplot(data = df, x = "loan_status", y = "loan_amnt")


# **Calculating the summary statistics for the loan amount, grouped by the loan_status.**

# In[77]:


df.groupby("loan_status")["loan_amnt"].describe()


# **exploring the Grade and SubGrade columns that LendingClub attributes to the loans.**

# In[88]:


x = df["grade"].unique()
print(x)


# In[106]:


y = df["sub_grade"].unique()
sorted(list(y))
print(y)


# **Creating a countplot per grade. Setting the hue to the loan_status label.**

# In[92]:


sns.countplot(data = df, x = "grade", hue = "loan_status")


# **Display a count plot per subgrade.**

# In[108]:


plt.figure(figsize = (12,5))
sns.countplot(data = df, x = df["sub_grade"],palette = "coolwarm",order= sorted(list(y)))


# In[111]:


plt.figure(figsize = (12,4))
sns.countplot(data = df, x = df["sub_grade"],palette = "coolwarm",hue = "loan_status",order= sorted(list(y)))


# **It looks like F and G subgrades don't get paid back that often. Isloating those and recreating the countplot just for those subgrades.**

# In[132]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# **Creating a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".**

# In[107]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[36]:


##
my_list = []
for item in df["loan_status"]:
    if (item == "Fully Paid"):
        my_list.append(1)
    else:
        my_list.append(0)
##


# In[136]:


df[["loan_repaid","loan_status"]]


# ---
# # Data PreProcessing
# 
# **Goals: Remove or fill any missing data. Remove unnecessary or repetitive features. Convert categorical string features to dummy variables.**

# In[137]:


df.head()


# # Missing Data
# 
# **Let's explore this missing data columns. I will use a variety of factors to decide whether or not they would be useful, to see if I should keep, discard, or fill in the missing data.**

# In[138]:


len(df)


# **Creating a Series that displays the total count of missing values per column.**

# In[5]:


df.isnull().sum()


# **Convert this Series to be in term of percentage of the total DataFrame**

# In[3]:


100* df.isnull().sum()/len(df)


# **Let's examine emp_title and emp_length to see whether it will be okay to drop them.**

# In[13]:


feat_info("emp_title")
print("\n")
feat_info("emp_length")


# In[17]:


df["emp_title"].nunique()


# In[18]:


df['emp_title'].value_counts()


# **Realistically there are too many unique job titles to try to convert this to a dummy variable feature.I will remove that emp_title column.**

# In[108]:


df = df.drop("emp_title",axis =1)


# **Create a count plot of the emp_length feature column.**

# In[31]:


plt.figure(figsize = (10,5))
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
sns.countplot(data = df, x = "emp_length", order= emp_length_order)


# **Plotting out the countplot with a hue separating Fully Paid vs Charged Off**

# In[32]:


plt.figure(figsize = (10,5))
sns.countplot(data = df, x = "emp_length", order= emp_length_order, hue = "loan_status")


# **This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan. There are a multitude of ways to create this Series.**

# In[10]:


df.head()


# In[9]:


emp_C = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_F = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_C/emp_F
emp_len


# In[16]:


emp_len.plot(kind='bar')


# **Charge off rates are extremely similar across all employment lengths.It is safe to drop the emp_length column.**

# In[109]:


df = df.drop("emp_length", axis =1)


# **Revisiting the DataFrame to see what feature columns still have missing data.**

# In[19]:


df.isnull().sum()


# In[22]:


df["purpose"].head(10)


# In[23]:


df['title'].head(10)


# **The title column is simply a string subcategory/description of the purpose column. I will drop the title column.**

# In[110]:


df = df.drop("title",axis =1)


# ---
# **REPLACING MISSING VALUES :Find out what the mort_acc feature represents**

# In[25]:


feat_info('mort_acc')


# In[26]:


df["mort_acc"].value_counts()


# **There are many ways we could deal with this missing data. I could attempt to build a simple model to fill it in, such as a linear model, I could just fill it in based on the mean of the other columns, or I could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc**

# In[32]:


x = df.corr()
x["mort_acc"]


# **Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. To get the result below:**

# In[39]:


tacc = df.groupby("total_acc").mean()["mort_acc"]
tacc
(tacc)


# **Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns.** (This is a very important step) 

# In[81]:


''''''
for items in df["mort_acc"]:
    i = items
    if items = NA:
        items = tacc.iloc[]
''''


# In[40]:


def fill(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return tacc[total_acc]
    else:
        return mort_acc


# In[111]:


df["mort_acc"] = df.apply(lambda x: fill(x['total_acc'], x['mort_acc']), axis=1)


# In[51]:


df.isnull().sum()


# **revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data.I will  go ahead and remove the rows that are missing those values in those columns.**

# In[112]:


df = df.dropna(axis=0)


# In[54]:


df.isnull().sum()


# ## Categorical Variables and Dummy Variables
# 
# **We're done working with the missing data! Now we just need to deal with the string values due to the categorical columns.**

# In[76]:


df.select_dtypes(['object']).columns


# ---
# **Let's now go through all the string features to see what we should do with them.**
# 
# ---
# 
# 
# ### term feature
# 
# **Converting the term feature into either a 36 or 60 integer numeric data type.**

# In[113]:


#
df["term"] = df["term"].apply(lambda x: int(x[:3]))


# ### grade feature
# 
# **We already know grade is part of sub_grade, so I will just drop the grade feature.**

# In[114]:


df.drop("grade",axis = 1)


# **Converting the subgrade into dummy variables. Then I will concatenate these new columns to the original dataframe.**

# In[115]:


dummies = pd.get_dummies(data = df["sub_grade"],drop_first = True)


# In[120]:


df_2 = pd.concat([df,dummies],axis=1)
df_2.columns


# In[121]:


df_2.drop("sub_grade",axis =1)


# ### verification_status, application_type,initial_list_status,purpose 
# **Converting these columns: ['verification_status', 'application_type','initial_list_status','purpose'] into dummy variables and concatenating them with the original dataframe.**

# In[124]:


dummies_2 = pd.get_dummies(data = df[['verification_status', 'application_type','initial_list_status','purpose']],drop_first=True)


# In[125]:


df_3 = pd.concat([df_2,dummies_2],axis =1)


# In[129]:


df_3 = df_3.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)


# ### home_ownership

# In[133]:


df_3["home_ownership"].value_counts()


# **Converting these to dummy variables, and replace NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe.**

# In[135]:


df_3["home_ownership"].replace("NONE", "OTHER", inplace=True)
df_3["home_ownership"].replace("ANY", "OTHER", inplace=True)
df_3["home_ownership"].unique()


# In[148]:


dummies_3 = pd.get_dummies(df["home_ownership"], drop_first =True)
df_4 = pd.concat([df_3,dummies_3],axis=1)
df_4 = df_4.drop(["home_ownership","grade","sub_grade"],axis=1)


# ### address
# **Let's feature engineer a zip code column from the address in the data set. Creating a column called 'zip_code' that extracts the zip code from the address column.**

# In[177]:


# last item of list is zipcode 
df_4["zip_code"] = df_4["address"].apply(lambda x: x.split()[-1])
df_4.head()


# **Now making this zip_code column into dummy variables using pandas. Concatenating the result and drop the original zip_code column along with dropping the address column.**

# In[229]:


dummy_zip = pd.get_dummies(df_4["zip_code"])
df_5 = pd.concat([df_4,dummy_zip],axis =1)
df_5 = df_5.drop("address",axis=1)


# ### issue_d 
# 
# **This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, dropping this feature.**

# In[230]:


df_5 = df_5.drop("issue_d",axis=1)


# In[231]:


df_5.loc[1,]
df_5.size


# ### earliest_cr_line
# **This appears to be a historical time stamp feature. I will Extract the year from this feature, then convert it to a numeric feature. Setting this new data to a feature column called 'earliest_cr_year'.**

# In[232]:


df_5['earliest_cr_year'] = df_5['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df_5 = df_5.drop('earliest_cr_line',axis=1)


# In[233]:


df_5.select_dtypes(['object']).columns


# ## FINALLY :Train Test Split

# In[236]:


from sklearn.model_selection import train_test_split


# **dropping the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.**

# In[282]:


df_5 = df_5.drop("loan_status",axis=1)
df_5 = df_5.drop("zip_code",axis = 1)


# In[242]:


df_5["loan_repaid"].unique()


# **Set X and y variables to the .values of the features and label.**

# In[283]:


X = df_5.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[121]:


# df = df.sample(frac=0.1,random_state=101)
print(len(df))


# **test_size=0.2 and a random_state of 101.**

# In[284]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# ## Normalizing the Data
# 
# **Using a MinMaxScaler to normalize the feature data X_train and X_test.**

# In[250]:


from sklearn.preprocessing import MinMaxScaler


# In[251]:


scaler = MinMaxScaler()


# In[285]:


X_train = scaler.fit_transform(X_train)


# In[286]:


X_test = scaler.transform(X_test)


# # Creating the Model
# 

# In[254]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# **Building a sequential model to will be trained on the data.**

# In[287]:


# CODE HERE
model = Sequential()

# Input layer
model.add(Dense(78, activation = "relu"))
model.add(Dropout(0.2))

# Hidden
model.add(Dense(39, activation = "relu"))
model.add(Dropout(0.2))

#Hidden
model.add(Dense(19, activation = "relu"))
model.add(Dropout(0.2))

#output
model.add(Dense(1, activation = "relu"))
model.add(Dropout(0.2))
# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


# **Fitting the model to the training data for at least 25 epochs. Also adding in the validation data for later plotting. add in a batch_size of 256.**

# In[288]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )


# In[289]:


from tensorflow.keras.models import load_model
model.save('Neural_Network_model.h5')  


# # Section 3: Evaluating Model Performance.
# 
# **Plotting out the validation loss versus the training loss.**

# In[290]:


losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()


# **Creating predictions from the X_test set and display a classification report and confusion matrix for the X_test set.**

# In[295]:


from sklearn.metrics import classification_report,confusion_matrix
predictions = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test,predictions))


# In[297]:


(confusion_matrix(y_test,predictions))


# **FINAL TEST: Given the customer information below, would I offer this person a loan?**

# In[321]:


import random
random.seed(101)
random_ind = random.randint(0,len(df_5))

new_customer = df_5.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[322]:


len(new_customer)


# **MY model tells me that it is 92% confident about giving a loan to this new customer**

# In[326]:


model.predict(new_customer.values.reshape(1,81))


# **Result: I decided to give the customer a loan and I will Now check if this person actually end up paying back their loan?**

# In[323]:


df.iloc[random_ind]['loan_repaid']


# **My database shows that the person actually did pay his loan back, thus the model was, overall, a success!**
# 
# #END OF PROJECT
