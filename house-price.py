#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings('ignore')



# In[3]:


def two_d_compare(X_test, y_test, y_pred, model_name):
    plt.subplots(ncols=2, figsize=(10, 4))
    plt.suptitle('Actual vs Predicted data : ' + model_name + '. Variance score: %.2f' % r2_score(y_test, y_pred))

    plt.subplot(121)
    plt.scatter(X_test, y_test, alpha=0.8, color='#8CCB9B')
    plt.title('Actual')

    plt.subplot(122)
    plt.scatter(X_test, y_pred,alpha=0.8, color='#E5E88B')
    plt.title('Predicted')

    plt.show()


def model_metrics(regressor, y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    print("Root mean squared error: %.2f" % math.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    print('R2 score: %.2f' % r2)
    return [mse, r2]
    

def two_vs_three(x_test,y_test,y_pred,z=None, isLinear = False):
    area = 60

    fig = plt.figure(figsize=(6,6))
    fig.suptitle('view of sales price data')

    # First subplot
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_test, y_test, alpha=0.5,color='blue', s= area)
    ax.scatter(x_test, y_pred, alpha=0.5,color='red', s= area)
#     ax.plot(x_test, y_pred, alpha=0.9,color='red', linewidth=2)
    ax.set_xlabel('YEARBUILT')
    ax.set_ylabel('SELLING PRICE')
    
    plt.title('YEARBUILT vs SALEPRICE')

    plt.show()
    


# In[4]:


df_pd =  pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/O9xijcDVZM-5MCjMyPtxwQ/house-price.csv")
df_pd.head()


# In[5]:


df_pd.columns


# In[6]:


df_pd.tail


# In[7]:


area = 60
x = df_pd['YEARBUILT']
y = df_pd['SALEPRICE']
z = df_pd['LOTAREA']


fig = plt.figure(figsize=(12,6))
fig.suptitle('2D and 3D view of sales price data')

# First subplot
ax = fig.add_subplot(1, 2, 1)

ax.scatter(x, y, alpha=0.5,color='blue', s=area)
ax.set_xlabel('YEAR BUILT')
ax.set_ylabel('SELLING PRICE')

plt.title('YEARBUILT vs SALEPRICE')

# Second subplot
ax = fig.add_subplot(1,2,2, projection='3d')

ax.scatter(z, x, y, color='blue', marker='o')

ax.set_ylabel('YEAR BUILT')
ax.set_zlabel('SELLING PRICE')
ax.set_xlabel('LOT AREA')

plt.title('LOT AREA VS YEAR BUILT vs SELLING PRICE')

plt.show()


# In[8]:


sns.set(rc={"figure.figsize": (8, 4)})
np.random.seed(0)
ax = sns.distplot(df_pd['SALEPRICE'])
plt.show()


# In[9]:


print("The dataset contains columns of the following data types : \n" +str(df_pd.dtypes))


# In[10]:


print("The dataset contains following number of records for each of the columns : \n" +str(df_pd.count()))


# In[11]:


df_pd.isnull().any()


# In[12]:


df_pd = df_pd.drop(['ID'], axis=1)

df_pd.head()


# In[13]:


categoricalColumns = df_pd.select_dtypes(include=['object']).columns

print("Categorical columns: ")
print(categoricalColumns)

impute_categorical = SimpleImputer(strategy='most_frequent')
onehot_categorical = OneHotEncoder(handle_unknown='ignore')

categorical_transformer = Pipeline(steps=[('impute', impute_categorical),
                                          ('onehot', onehot_categorical)])


# In[14]:


numericalColumns = [col for col in df_pd.select_dtypes(include=['float', 'int']).columns if col not in ['SALEPRICE']]
print("Numerical columns: ")
print(numericalColumns)

scaler_numerical = StandardScaler()

numerical_transformer = Pipeline(steps=[('scale', scaler_numerical)])


# In[15]:


preprocessorForCategoricalColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns)],
                                            remainder="passthrough")
preprocessorForAllColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns),
                                                            ('num',numerical_transformer,numericalColumns)],
                                              remainder="passthrough")


# In[16]:


df_pd_temp = preprocessorForCategoricalColumns.fit_transform(df_pd)
print("Data after transforming :")
print(df_pd_temp)

df_pd_temp_2 = preprocessorForAllColumns.fit_transform(df_pd)
print("Data after transforming :")
print(df_pd_temp_2)


# In[17]:


features = []
features = df_pd.drop(['SALEPRICE'], axis=1)

label = pd.DataFrame(df_pd, columns=['SALEPRICE'])
label = np.log(df_pd['SALEPRICE'])

print("value of label: \n" + str(label))


# In[18]:


X = features['YEARBUILT'].values.reshape(-1,1)
X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(X, label, random_state=0)

print("Dimensions of datasets that will be used for training: Input features"
      + str(X_train_slr.shape)
      + ". Output label" + str(y_train_slr.shape))
print("Dimensions of datasets that will be used for testing: Input features"
      + str(X_test_slr.shape)
      + ". Output label"
      + str(y_test_slr.shape))


# In[19]:


from sklearn.linear_model import LinearRegression

model_name = 'Simple Linear Regression'

slRegressor = LinearRegression()

slRegressor.fit(X_train_slr, y_train_slr)

y_pred_slr = slRegressor.predict(X_test_slr)

print(slRegressor)


# In[20]:


print('Intercept: \n',slRegressor.intercept_)
print('Coefficients: \n', slRegressor.coef_)


# In[21]:


two_vs_three(X_test_slr[:, 0], y_test_slr, y_pred_slr, None, True)


# In[22]:


two_d_compare(X_test_slr, y_test_slr, y_pred_slr, model_name)


# In[23]:


slrMetrics = model_metrics(slRegressor,y_test_slr,y_pred_slr)


# In[ ]:




