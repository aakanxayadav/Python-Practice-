#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd


# In[8]:


s= pd.Series([1,2,3,"a","b","abc"])
s


# In[9]:


s1=pd.Series([1,2,3,4,5,6], index=["a","b","c","d","e","f"])
s1


# In[10]:


s4=pd.Series(["ABC","HFDBG","JDGT","GDR","HSF"])
s4.str.lower()


# In[11]:


s4[0]


# In[12]:


s4[:3]


# In[13]:


s4[0:5]


# In[14]:


s4[:-1]


# In[15]:


s1[s1>s1.median()]


# In[16]:


s1[s1<s1.median()]


# In[17]:


s4[[2,4,1]]


# In[18]:


s.dtype


# In[19]:


s5=s1+s4
s5


# In[20]:


"e" in s1


# In[21]:


"z" in s1


# In[22]:


s1[1:]+s1[:-1]


# In[23]:


e={
    "name":pd.Series(["jay","sree","ram","kallu","jani"],index=["a","b","c","d","e"]),
    "cont no":pd.Series(["123456789","987654321","614259825","168344256","662451835"],index=["a","b","c","d","e"]),
    "age":pd.Series([20,30,40,25,19],index=["a","b","c","d","e"])
}
df=pd.DataFrame(e)
df


# In[24]:


e={"name":["jay","sree","ram","kallu","jani"],"cont no":["123456789","987654321","614259825","168344256","662451835"],
    "age":[20,30,40,25,19]
}
df=pd.DataFrame(e)
df


# In[25]:


import pandas as pd

data = [("jay", "123456789", 20),
        ("sree", "987654321", 30),
        ("ram", "614259825", 40),
        ("kallu", "168344256", 25),
        ("jani", "662451835", 19)]

c = ["name", "cont no", "age"]

df = pd.DataFrame(data, columns=c)
df


# In[26]:


df.columns


# In[27]:


df.T


# In[28]:


df.sort_index(axis = 0,ascending = True)


# In[29]:


df.sort_values(by = "age")


# In[30]:


df["age"]


# In[31]:


df.loc[0]


# In[32]:


df.loc[0:1,"age"]


# In[33]:


df[df["age"]>19]


# In[34]:


df[df["age"].isin([19])]


# In[36]:


import pandas as pd

data = [("jay", 12, 13, 16, 17),
        ("sree", 13, 15, 15, 25),
        ("ram", 9, 18, 12, 20),
        ("kallu", 12, 11, 13, 25),
        ("jani", 12, 15, 16, 19)]

c = ["name", "bda", "sda", "tsfta", "eimi"]  # Corrected column names

df = pd.DataFrame(data, columns=c)
print(df)


# In[37]:


left = pd.DataFrame({"key":["a","b","c"],"LVal":[1,2,3]})
right = pd.DataFrame({"key":["a","b","c"],"LVal":[4,5,6]})


# In[38]:


pd.merge(left,right,on="key")


# In[39]:


df6=pd.DataFrame({"id": [1,2,3,4,5,6], "Raw_Grade": ["A","B","E","E","B","A"]})


# In[42]:


import pandas as pd

# Sample DataFrame
data = {
    "Raw_Grade": ["A", "B", "C", "A", "D", "B", "C", "A", "B", "A"]
}
df6 = pd.DataFrame(data)

# Convert 'Raw_Grade' to categorical
df6["grade"] = df6["Raw_Grade"].astype("category")

# Assigning new categories
df6["grade"] = df6["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])

# Sorting the data
df6_sorted = df6.sort_values(by="grade")

# Grouping
grouped = df6_sorted.groupby("grade").size()

print(df6_sorted)
print(grouped)


# In[43]:


from sklearn.datasets import load_iris
#load iris dataset
iris=load_iris()
data=pd.DataFrame(data=iris.data,columns=iris.feature_names)
data['target']=iris.target_names[iris.target]
data.head()


# # DATA EXPLORATION

# In[6]:


data.info()


# In[56]:


data.isnull()


# In[57]:


data.describe()


# # DATA SELECTION

# In[39]:


selected_columns = [ 'sepal length (cm)', 'sepal width (cm)']

new_df = data[selected_columns]
new_df.head()


# In[58]:


result = data[data['petal length (cm)'] > 4.5]
print(result)


# # DATA MANIPULATION

# In[60]:


data['sepal area'] = data['sepal length (cm)'] * data['sepal width (cm)']

# Print the DataFrame
data.head()


# In[41]:


grade_mapping = {'setosa': 1, 'versicolor':2, 'virginica': 3}
data["target"] = data["target"].map(grade_mapping)
data.head()


# # DATA ANALYSIS

# In[42]:


petal_length_mean = data.groupby("target")["petal length (cm)"].mean()
print(petal_length_mean)


# In[43]:


data["sepal_area"] = data["sepal length (cm)"] * data["sepal width (cm)"]
max_sepal_area = data["sepal_area"].max()
print("The maximum sepal area in the dataset is:", max_sepal_area, "cm^2")


# # DATA FILTERING

# In[46]:


filtered_df = data[(data['sepal length (cm)'] < 5) & (data['petal width (cm)'] > 1)]
filtered_df.head()


# # DATA AGGREGATION

# In[51]:


petal_length_max = data.groupby("target")["petal length (cm)"].max()
print(petal_length_max)


# In[53]:


sepal_width_min = data.groupby("target")["sepal width (cm)"].min()
print(sepal_width_min)
sepal_width_max = data.groupby("target")["sepal width (cm)"].max()
print(sepal_width_max)


# # Data Visualisation

# In[47]:


plt.figure(figsize=(8, 6))
plt.hist(data['petal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Petal Length (cm)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[49]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=data, palette='viridis')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.grid(True)
plt.show()


# # Data Merging

# In[51]:


data_random = pd.DataFrame(np.random.rand(len(data)), columns=['random_values'])
data_merged = pd.concat([data, data_random], axis=1)
data_merged


# # Data Grouping

# In[52]:


mean_values_by_target = data.groupby('target').mean()
mean_values_by_target


# # Data Pivot

# In[53]:


pivot_table_mean_sepal_length = data.pivot_table(index='target', values='sepal length (cm)', aggfunc='mean')
pivot_table_mean_sepal_length


# # Data transformation

# In[54]:


from sklearn.preprocessing import MinMaxScaler

numerical_columns = data.drop('target', axis=1)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numerical_columns)
normalized_df = pd.DataFrame(normalized_data, columns=numerical_columns.columns)

normalized_df['target'] = data['target']
normalized_df


# # Data Handling

# In[55]:


from sklearn.preprocessing import MinMaxScaler

numerical_columns = data.drop('target', axis=1)

target_column = data['target']

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(numerical_columns)

normalized_df = pd.DataFrame(normalized_data, columns=numerical_columns.columns)

normalized_df['target'] = target_column

normalized_df


# In[ ]:




