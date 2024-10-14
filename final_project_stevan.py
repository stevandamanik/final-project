

#preprocessing
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from imblearn import over_sampling
from sklearn.preprocessing import LabelEncoder

#modelling
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV


#metrik
from sklearn.metrics import accuracy_score

# import dalex to explain complex model
import dalex as dx

# load shap package for shap explanation
import shap


# streamlit app layout
st.title('Digital Marketing Insight')
st.write("""
Aplikasi ini akan memberikan informasi mengenai media digital marketing yang efektif
""")

url = "digital_marketing_campaign_dataset.csv"
pd.set_option('display.max_columns', None)
data = pd.read_csv(url)

#check missing values
data.isnull().sum()

#cek duplicated
data['CustomerID'].duplicated().sum()

data = data.drop('CustomerID',axis = 1)

##Cek Outlier

num_data = data.select_dtypes(['float64','int64'])
cat_data = data.select_dtypes('object')

data_new = data.copy()

#drop feature yang tidak memberikan pengaruh dianalisis, sehingga tidak perlu di encode
data_new = data.drop(['AdvertisingPlatform','AdvertisingTool'],axis = 1)

data_new =pd.get_dummies(data=data_new,columns=['Gender','CampaignChannel','CampaignType'])

column = data_new.select_dtypes('bool').columns.to_list()


data_new[column] = data_new[column].astype('int64')


#Split Data

feature =  data_new.drop(columns='Conversion')
target = data_new['Conversion']

# train and val
X_pretrain, X_test, y_pretrain, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)


#train and validation
X_train, X_validation, y_train, y_validation = train_test_split(X_pretrain, y_pretrain,test_size = 0.20, random_state = 42)

#X_train.shape,X_validation.shape,X_test.shape

#Handling Imbalance data

X_over_smote, y_over_smote = over_sampling.SMOTE().fit_resample(X_train, y_train)

y_over_smote.value_counts()

#Membangun model

#model xgboost
model_xgb = xgb.XGBClassifier(learning_rate = 0.3,
                              max_depth = 3,
                              n_estimators = 300,
                              subsample = 0.8,
                              random_state = 42)
model_xgb.fit(X_over_smote, y_over_smote)
y_pred_xgb = model_xgb.predict(X_test)


#Evaluasi

accuracy = accuracy_score(y_test, y_pred_xgb)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

vis_option = st.selectbox(
    'Pilih jenis visualisasi:',
    ['Distribusi Gender', 'Campaign Channel Paling Efektif', 'Feature Importance']
)

if vis_option =='Distribusi Gender':
   fig, ax = plt.subplots(5,3,figsize=(10,15))
   
   gender = data['Gender'].value_counts()
   gender_label = gender.index.tolist()  # Pastikan hanya label gender
   
   colors = ['#00ffff', '#66b3ff']
   
   # Membuat plot pie chart
   fig, ax = plt.subplots(figsize=(5, 5))
   my_circle = plt.Circle((0, 0), 0.7, color='white')
   ax.pie(gender, labels=gender_label, colors=colors, wedgeprops={'linewidth': 5, 'edgecolor': 'white'}, autopct='%1.1f%%')
   plt.gca().add_artist(my_circle)
   plt.title('Segment Market', weight='bold')
   st.pyplot(fig)

elif vis_option == 'Campaign Channel Paling Efektif':
  fig = plt.figure(figsize=(10, 6))
  
  sns.barplot(x='CampaignChannel', y='AdSpend', data=data,estimator='mean',palette='viridis')
  plt.xlabel("Campaign Channel")
  plt.ylabel("Ad Spend")
  plt.title("Biaya Ads Setiap Channel")
  plt.xticks(rotation=45)

  st.pyplot(fig)
  

elif vis_option == 'Feature Importance':
  # Plot Feature Importances
  importance = pd.Series(model_xgb.feature_importances_, index=X_test.columns)
  viz_imp = importance.plot(kind='barh')

  top_option = st.radio("Tampilkan:", ("Semua Features", "Top 5 Features"))
  if top_option == "Semua Features":
     importance = pd.Series(model_xgb.feature_importances_, index=X_test.columns)
     viz_imp = importance.plot(kind='barh')
     st.pyplot(viz_imp.figure)

  if top_option == "Top 5 Features":
        importance = importance.nlargest(5)
        viz_imp = importance.plot(kind='barh')
        st.pyplot(viz_imp.figure)
