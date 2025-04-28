<H3>P PARTHIBAN</H3>
<H3>E212223230145</H3>
<H3>EX. NO.1</H3>
<H3>25.04.2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
TYPE YOUR CODE HERE
```python
#import libraries
import numpy as np
from google.colab import files
import pandas as pd
import io
from sklearn.impute import SimpleImputer as sim  
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
dataset = pd.read_csv("/content/Churn_Modelling (1).csv")
print(dataset.head())
print(dataset.tail())

# Finding Missing Values
print("Missing Values: \n ",dataset.isnull().sum())

# hadling the missing dataset's using the mean in number column
impute_txt = sim(strategy='mean')
text_col = dataset.select_dtypes(include=[np.number]).columns
dataset[text_col] = impute_txt.fit_transform(dataset[text_col])

#Check for Duplicates
print("Duplicate values:\n ")
print(dataset.duplicated())

# encoding the categorical data (Text) into number (since ml model can't understand the missing data)
label = {}
for i in text_col:
    le = LabelEncoder()
    dataset[i] = le.fit_transform(dataset[i])
    label[i] = le

# Normalizing the data to perform algorithm's better
scale = StandardScaler()
dataset[text_col] = scale.fit_transform(dataset[text_col])
print("Normalized data for column(s):", text_col)
print(dataset[text_col])

# splitting the datasets to train the model     
x = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Print the training data and testing data
print("Training data")
print(x_train)
print(y_train)

print("Testing data")
print(x_test)
print(y_test)
print("Length of X_test: ", len(x_test))

```

## OUTPUT:

### df.head(): 
![image](https://github.com/user-attachments/assets/84293347-5a8b-4ab4-9645-5e89ddbc06a2)

### df.tail():
![image](https://github.com/user-attachments/assets/a915296f-541a-430b-8629-0343a6e483fd)

### Missing Values:
![image](https://github.com/user-attachments/assets/9dc1c689-33e6-4070-bf3a-9d11c2777f20)

### Duplicate values:
![image](https://github.com/user-attachments/assets/bb4d54bb-c2ca-4f2d-8e83-522463ec1b4a)

### Normizing Data:
![image](https://github.com/user-attachments/assets/5a0a8767-e8a5-4939-8179-26c933a97372)

### Training data:
![image](https://github.com/user-attachments/assets/9cea9888-6341-46a9-837c-0c3338040cd6)

### Testing data:
![image](https://github.com/user-attachments/assets/7c46e792-6924-42f8-ab01-e330f741a3ec)

### Length of X_test
![image](https://github.com/user-attachments/assets/b7310797-d7f8-482f-9ddc-f460185025c1)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


