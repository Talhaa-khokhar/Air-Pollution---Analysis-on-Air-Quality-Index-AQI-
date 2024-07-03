#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


# In[2]:


data = pd.read_csv('Cleaned.csv')


# # Preprocessed Dataset but just Checking

# In[3]:


data


# In[4]:


rows, cols = data.shape
print("The dataset contains",rows,"rows and",cols,"columns" )


# In[5]:


data.info()


# In[6]:


data.head()


# In[7]:


data.tail(10)


# In[8]:


data.tail()


# In[9]:


data.size


# In[10]:


data.columns


# In[11]:


data.duplicated().sum()


# In[12]:


data.nunique()


# In[13]:


data.isnull().sum()


# In[14]:


len(data.columns)


# In[15]:


data.shape


# # Descriptive statistics:

# In[16]:


data.describe()


# In[17]:


Descriptive_statistics = data.describe()
print(Descriptive_statistics)


# # Distribution analysis: Histograms.

# In[18]:


numeric_cols = data.select_dtypes(include=[int, float]).columns


# In[19]:


#histograms
for col in numeric_cols:
    plt.figure(figsize=(10, 12))
    sns.histplot(data[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.show()


# # Distribution analysis: Box plots.

# In[20]:


#box plot
for col in numeric_cols:
    plt.figure(figsize=(10, 12))
    sns.boxplot(data[col])
    plt.title(f'Box Plot of {col}')
    plt.show()


# # If we talk about country then count plot for just country.

# In[21]:


#count plot
plt.figure(figsize=(10, 12))
sns.countplot(x='Country', data=data)
plt.title('Count Plot of Country')
plt.show()


# # Correlation analysis: Heatmaps to identify relationships between variables.

# In[22]:


corr_matrix = data.corr()
#heatmap
plt.figure(figsize=(16, 18))
sns.heatmap(corr_matrix, annot=True, cmap='cubehelix', square=True)
plt.title('Correlation Heatmap')
plt.show()


# # Show trends over time

# In[23]:


plt.figure(figsize=(14, 7))
plt.plot(data.groupby('City')['AQI Value'].mean().index, data.groupby('City')['AQI Value'].mean().values)
plt.xlabel('City')
plt.ylabel('AQI Value')
plt.title('Trend of AQI Value over Time')
plt.show()


# In[24]:


plt.figure(figsize=(14, 7))
for country in data['Country'].unique():
    country_data = data[data['Country'] == country]
    plt.plot(country_data.groupby('City')['AQI Value'].mean().index, country_data.groupby('City')['AQI Value'].mean().values, label=country)
plt.xlabel('City')
plt.ylabel('AQI Value')
plt.title('Trend of AQI Value over Time by Country')
plt.legend()
plt.show()


# # Bar charts and pie charts to show the distribution of AQI levels across different locations.
# 

# In[25]:


# Bar chart
plt.figure(figsize=(14, 7))
sns.countplot(x='City', hue='AQI Category', data=data)
plt.title('Distribution of AQI Levels by City')
plt.show()


# In[26]:


# pie charts 
plt.figure(figsize=(14, 7))
counts = data['Country'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%')
plt.title('Distribution of AQI Levels by Country')
plt.show()


# # Scatter plots

# In[56]:


# Scatter plots for AQI Value vs. other pollutants
for col in ['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']:
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=col, y='AQI Value', data=data)
    plt.title(f'Relationship between {col} and AQI Value')
    plt.show()


# In[76]:


get_ipython().system('pip install geopandas')


# In[77]:


import geopandas as gpd


# In[79]:


# Loading a world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge AQI data with the map data based on city names or codes
merged = world.merge(data, how='left', left_on='name', right_on='City')

#Choropleth map of PM2.5 AQI
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(column='PM2.5 AQI Value', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('PM2.5 AQI Distribution')
ax.set_axis_off()
plt.show()

#Choropleth map of Ozone AQI
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(column='Ozone AQI Value', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Ozone AQI Distribution')
ax.set_axis_off()
plt.show()

#Choropleth map of CO AQI
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(column='CO AQI Value', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('CO AQI Distribution')
ax.set_axis_off()
plt.show()

#Choropleth map of NO2 AQI
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
merged.plot(column='NO2 AQI Value', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('NO2 AQI Distribution')
ax.set_axis_off()
plt.show()


# In[ ]:





# # Statistical Analysis:
# 

# # Hypothesis testing: T-tests, chi-square tests

# In[66]:


city_a_aqi = data[data['City'] == 'City A']['AQI Value']
city_b_aqi = data[data['City'] == 'City B']['AQI Value']

if not city_a_aqi.empty and not city_b_aqi.empty:
    t_stat, p_val = ttest_ind(city_a_aqi, city_b_aqi)
    print(f'T-statistic: {t_stat}, P-value: {p_val}')
else:
    print("One or both of the city AQI values are empty.")


# In[30]:


aqi_categories = data['AQI Category']
cities = data['City']

contingency_table = pd.crosstab(aqi_categories, cities)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f'Chi-square statistic: {chi2}, p-value: {p}')


# # Regression analysis: 

# # Linear regression

# # Test & Train

# In[31]:


X = data['CO AQI Value']
y = data['AQI Value']


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[33]:


model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)


# In[34]:


print(f'Coefficient: {model.coef_[0]}, R-squared: {model.score(X_test.values.reshape(-1, 1), y_test)}')


# # Multiple regression

# # Test and Train

# In[35]:


X = data[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = data['AQI Value']


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[37]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[38]:


print(f'Coefficients: {model.coef_}, R-squared: {model.score(X_test, y_test)}')


# # k-means clustering.

# In[39]:


X = data[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]


# In[40]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)


# In[41]:


print(f'Cluster labels: {kmeans.labels_}')


# In[42]:


k_values = range(2, 9)
inertia_values = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

                # Determine the optimal K based on the Elbow method
optimal_k = 4  # or the value you determine from the plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)


# # Machine learning models:

# # Decision tree

# # Train and Test

# In[43]:


X = data[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = data['AQI Value']


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[45]:


model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[46]:


model.fit(X_train, y_train)


# In[47]:


y_pred = model.predict(X_test)


# In[48]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[49]:


print(f'Decision Tree:')
print(f'MSE: {mse:.3f}')
print(f'R-squared: {r2:.3f}')


# # support vector machines

# # Train and Test

# In[50]:


X = data[['CO AQI Value', 'Ozone AQI Value', 'NO2 AQI Value', 'PM2.5 AQI Value']]
y = data['AQI Value']


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[52]:


model = SVR(kernel='rbf', C=1, epsilon=0.1)
model.fit(X_train, y_train)


# In[53]:


y_pred = model.predict(X_test)
print(f'R-squared: {model.score(X_test, y_test)}')


# In[67]:


# Conclusions
print('Conclusions:')
print('------------')
print('Based on the analysis, we found that:')
print('* AQI values are highly correlated with CO AQI Value, Ozone AQI Value, and PM2.5 AQI Value.')
print('* The Decision Tree Regressor model performed best in predicting AQI values.')
print('* The SVR model performed poorly, indicating that it may not be suitable for this dataset.')

# Recommendations
print('Recommendations:')
print('--------------')
print('Based on the findings, we recommend:')
print('* Implementing policies to reduce CO, Ozone, and PM2.5 emissions in urban areas.')
print('* Increasing public awareness about the importance of air quality and its impact on health.')
print('* Conducting further research to identify other factors that contribute to air pollution.')


# In[ ]:




