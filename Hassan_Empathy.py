#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 (Empathy)

# In[1]:


# Importing the Required Libraries and looping through the csv files and concat them into one singe dataframe
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
from sklearn import preprocessing

path = r'EyeT'
data = []

#Looping through the files in the directory
#There are 512 csv files which contains the data we need to merge, os package helps us to identify .csv files
#and using the package we can read all csv one by one and append to create a single csv
for file in os.listdir(path):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join(path, file), low_memory=True)
        data.append(df)
df = pd.concat(data, ignore_index=True)


# In[2]:


df.to_csv('Final.csv')


# # Data Merging and saving is completed

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
from sklearn import preprocessing


# In[2]:


df = pd.read_csv('Final.csv')


# In[3]:


empathy_scoreA = pd.read_csv(r"C:\Users\HASSAN\Questionnaire_datasetIA.csv", encoding= 'unicode_escape') 
empathy_scoreB = pd.read_csv(r"C:\Users\HASSAN\Questionnaire_datasetIB.csv", encoding= 'unicode_escape') 


# In[4]:


df.head()


# In[5]:


empathy_scoreA.head()


# In[6]:


empathy_scoreB.head()


# In[7]:


# Checking for the percentage of null values of each column
for i in df.columns:
    print("Columns is:- ",i," And its null values are:- ",df[i].isnull().sum()/df.shape[0]*100)


# In[8]:


# Dropping those unnessessary columns and columns which has more than 30% of NaN values
df.drop(columns=['Export date','Recording date','Recording date UTC',
 'Recording start time',
 'Recording start time UTC',
 'Recording duration',
 'Recording software version',
 'Recording resolution height',
 'Recording resolution width',
 'Recording monitor latency','Recording timestamp','Presented Stimulus name',
'Presented Media name','Computer timestamp','Participant name','Recording name',
'Timeline name','Recording Fixation filter name','Unnamed: 0',
 'Event','Event value','Pupil diameter left','Pupil diameter right','Fixation point X',
 'Fixation point Y',
 'Fixation point X (MCSnorm)',
 'Fixation point Y (MCSnorm)',
 'Mouse position X',
 'Mouse position Y','Eyetracker timestamp','Presented Media height',
'Presented Media position Y (DACSpx)'],inplace=True)


# In[9]:


# Filling these column's NaN values to Not Recorded so that we can Do preprocessing via Label Encoders
df['Validity left'].fillna('Not Recorded', inplace=True)
df['Validity right'].fillna('Not Recorded', inplace=True)
df['Sensor'].fillna('Not Recorded', inplace=True)


# In[10]:


df.drop('Unnamed: 0.1',axis = 1,inplace = True)


# In[11]:


df.head()


# In[12]:


df['Eye movement type'].unique()


# In[13]:


import gc

gc.collect()

#Replacing the , to . for these columns and convert their type from object to float
string_replace = ['Eye position left X (DACSmm)',
 'Eye position left Y (DACSmm)',
 'Eye position left Z (DACSmm)',
 'Eye position right X (DACSmm)',
 'Eye position right Y (DACSmm)',
 'Eye position right Z (DACSmm)',
 'Gaze point left X (DACSmm)',
 'Gaze point left Y (DACSmm)',
 'Gaze point right X (DACSmm)',
 'Gaze point right Y (DACSmm)',
 'Gaze point X (MCSnorm)',
 'Gaze point Y (MCSnorm)',
 'Gaze point left X (MCSnorm)',
 'Gaze point left Y (MCSnorm)',
 'Gaze point right X (MCSnorm)',
 'Gaze point right Y (MCSnorm)',
 'Gaze direction left X',
 'Gaze direction left Y',
 'Gaze direction left Z',
 'Gaze direction right X',
 'Gaze direction right Y',
 'Gaze direction right Z']

# Convert the type of the column from object to float
for i in df[string_replace]:
    df[i] = df[i].astype(str).str.replace(',', '.')
    df[i] = df[i].astype(float)
    print(i, " is done")


# In[14]:


gc.collect()

mean_columns = ['Gaze point X',
 'Gaze point Y',
 'Gaze point left X',
 'Gaze point left Y',
 'Gaze point right X',
 'Gaze point right Y',
 'Gaze direction left X',
 'Gaze direction left Y',
 'Gaze direction left Z',
 'Gaze direction right X',
 'Gaze direction right Y',
 'Gaze direction right Z',
 'Eye position left X (DACSmm)',
 'Eye position left Y (DACSmm)',
 'Eye position left Z (DACSmm)',
 'Eye position right X (DACSmm)',
 'Eye position right Y (DACSmm)',
 'Eye position right Z (DACSmm)',
 'Gaze point left X (DACSmm)',
 'Gaze point left Y (DACSmm)',
 'Gaze point right X (DACSmm)',
 'Gaze point right Y (DACSmm)',
 'Gaze point X (MCSnorm)',
 'Gaze point Y (MCSnorm)',
 'Gaze point left X (MCSnorm)',
 'Gaze point left Y (MCSnorm)',
 'Gaze point right X (MCSnorm)',
 'Gaze point right Y (MCSnorm)',
 'Gaze direction left X',
 'Gaze direction left Y',
 'Gaze direction left Z',
 'Gaze direction right X',
 'Gaze direction right Y',
 'Gaze direction right Z']

# Fill NaN values in each column with the mean of those column
for col in df[mean_columns]:
    mean_val = df[col].mean()
    df[col].fillna(mean_val, inplace=True)
    print(col, " is done")

df.dropna(inplace = True)

# Making final dataframe and remove the Test group experiment and Control group experiment

final_df = df[~(df['Project name'] == 'Test group experiment') & ~(df['Project name'] == 'Control group experiment')]

# Converting column to Label Encoders
label_converter = ['Sensor','Validity left','Validity right','Eye movement type','Project name']
le = preprocessing.LabelEncoder()
for i in label_converter:
    final_df[i] = le.fit_transform(final_df[i])

final_df.reset_index(drop = True,inplace = True)


# In[15]:


df.shape, final_df.shape


# In[16]:


final_df.head()


# In[17]:


final_df.describe()


# In[18]:


final_df.corr()


# In[19]:


import seaborn as sns


# In[20]:


# generate correlation matrix
corr_matrix = final_df.corr().transpose()

# set figure size
plt.figure(figsize=(10, 10))

# plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm')

# set title
plt.title('Correlation Heatmap')

# show the plot
plt.show()


# In[21]:


final_df.hist()
plt.figure(figsize=(20, 20))
plt.show()


# In[22]:


final_df.dtypes


# In[31]:


final_df.duplicated().sum()


# In[32]:


final_df.drop_duplicates(inplace = True)


# In[34]:


final_df.isnull().sum().sum()


# In[35]:


final_df.head()


# In[36]:


final_df['Sensor'].unique()


# In[38]:


final_df.drop('Sensor',axis = 1,inplace = True)


# In[37]:


final_df['Project name'].unique()


# # Splitting the data 

# In[39]:


#seperating dependent and independent variable x,y
X = final_df.drop('Eye movement type',axis = 1)
y = final_df['Eye movement type']


# In[40]:


from sklearn.model_selection import train_test_split

#splitting the data into training 67 % and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[41]:


X_train.shape,y_train.shape


# In[42]:


X_test.shape,y_test.shape


# In[43]:


gc.collect()


# # Model Building for classifiying the eye movement 

# # Naive Bayes

# In[48]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

classify_1 = GaussianNB()
classify_1.fit(X_train,y_train)


# In[49]:


pred1 = classify_1.predict(X_test)


# In[50]:


print(classification_report(y_test, pred1))


# # Logistic Regression LR

# In[44]:


from sklearn.linear_model import LogisticRegression

classify_2 = LogisticRegression()
classify_2.fit(X_train,y_train)


# In[45]:


pred2 = classify_2.predict(X_test)


# In[46]:


print(classification_report(y_test, pred2))


# # Empathy score

# In[4]:


empathy_scoreB.head()


# In[5]:


final_empathy_scoreA_columns = []
for i in empathy_scoreA.columns:
    if empathy_scoreA[i].dtypes != 'O':
        final_empathy_scoreA_columns.append(i)
        
final_empathy_scoreA = empathy_scoreA[final_empathy_scoreA_columns]
final_empathy_scoreB = empathy_scoreB[final_empathy_scoreA_columns]


# In[6]:


final_empathy_scoreA.drop('NR',axis = 1,inplace = True)
final_empathy_scoreB.drop('NR',axis = 1,inplace = True)


# In[7]:


final_empathy_scoreA.describe()


# In[8]:


#check null values
final_empathy_scoreA.isnull().sum().sum()


# In[9]:


#Check duplicates
final_empathy_scoreA.duplicated().sum()


# In[10]:


final_empathy_scoreA.head()


# In[11]:


final_empathy_scoreA.corr()


# In[12]:


# generate correlation matrix
corr_matrix = final_empathy_scoreA.corr().transpose()

# set figure size
plt.figure(figsize=(30, 30))

# plot heatmap
sns.heatmap(corr_matrix, cmap='coolwarm')

# set title
plt.title('Correlation Heatmap')

# show the plot
plt.show()


# # splitting data into training and testing for empathy score

# In[13]:


#making EmpathyA csv for training the empathy score
X1_train = final_empathy_scoreA.drop('Total Score extended',axis = 1)
y1_train = final_empathy_scoreA['Total Score extended']


# In[14]:


#making EmpathyB csv for testing the empathy score
X1_test = final_empathy_scoreB.drop('Total Score extended',axis = 1)
y1_test = final_empathy_scoreB['Total Score extended']


# # Model Building for predicting empathy score

# # Linear Regression

# In[15]:


from sklearn.linear_model import LinearRegression

predict_1 = LinearRegression()
predict_1.fit(X1_train,y1_train)


# In[16]:


score_predicted1 = predict_1.predict(X1_test)


# In[17]:


from sklearn.metrics import mean_squared_error, r2_score
import math


mse = mean_squared_error(y1_test, score_predicted1)
r2 = r2_score(y1_test, score_predicted1)


# In[18]:


print('Root mean square of LR for predicting empathy score is:- ',math.sqrt(mse))
print('R2 of LR for predicting empathy score is:- ',r2)


# # Random Forest RF

# In[19]:


from sklearn.ensemble import RandomForestClassifier

predict_2 = RandomForestClassifier()
predict_2.fit(X1_train,y1_train)


# In[20]:


score_predicted2 = predict_2.predict(X1_test)


# In[21]:


mse_1 = mean_squared_error(y1_test, score_predicted2)
r2_1 = r2_score(y1_test, score_predicted2)


# In[22]:


print('Root mean square of LR for predicting empathy score is:- ',math.sqrt(mse_1))
print('R2 of LR for predicting empathy score is:- ',r2_1)


# In[ ]:




