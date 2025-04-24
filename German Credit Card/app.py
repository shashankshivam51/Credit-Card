import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv", index_col=0)

    df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
    df['Checking account'] = df['Checking account'].fillna('unknown')
  
    df['Credit_per_Duration'] = df['Credit amount'] / df['Duration']
 
    median_credit = df['Credit amount'].median()
    df['Risk'] = np.where(df['Credit amount'] > median_credit, 1, 0)
    return df

df = load_data()

st.sidebar.header("Applicant Information")
def user_input_features():
    Age = st.sidebar.slider('Age', int(df.Age.min()), int(df.Age.max()), int(df.Age.mean()))
    Sex = st.sidebar.selectbox('Sex', df['Sex'].unique())
    Job = st.sidebar.slider('Job (0=unskilled, 3=highly skilled)', int(df.Job.min()), int(df.Job.max()), int(df.Job.mean()))
    Housing = st.sidebar.selectbox('Housing', df['Housing'].unique())
    Saving_accounts = st.sidebar.selectbox('Saving accounts', df['Saving accounts'].unique())
    Checking_account = st.sidebar.selectbox('Checking account', df['Checking account'].unique())
    Credit_amount = st.sidebar.slider('Credit amount', int(df['Credit amount'].min()), int(df['Credit amount'].max()), int(df['Credit amount'].mean()))
    Duration = st.sidebar.slider('Duration (months)', int(df['Duration'].min()), int(df['Duration'].max()), int(df['Duration'].mean()))
    Purpose = st.sidebar.selectbox('Purpose', df['Purpose'].unique())
    Credit_per_Duration = Credit_amount / Duration if Duration > 0 else 0
    data = {'Age': Age,
            'Sex': Sex,
            'Job': Job,
            'Housing': Housing,
            'Saving accounts': Saving_accounts,
            'Checking account': Checking_account,
            'Credit amount': Credit_amount,
            'Duration': Duration,
            'Purpose': Purpose,
            'Credit_per_Duration': Credit_per_Duration}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()


X = df.drop(['Risk'], axis=1)
y = df['Risk']


cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_encoded = encoder.fit_transform(X[cat_cols])
X_num = X.drop(cat_cols, axis=1).values
X_all = np.hstack([X_encoded, X_num])


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_all, y)


input_encoded = encoder.transform(input_df[cat_cols])
input_num = input_df.drop(cat_cols, axis=1).values
input_all = np.hstack([input_encoded, input_num])

prediction = rf.predict(input_all)[0]
prediction_proba = rf.predict_proba(input_all)[0][1]

st.title("German Credit Risk Prediction")
st.write("Enter applicant details in the sidebar to predict credit risk.")

st.subheader("Prediction")
st.write("**Credit Risk:**", "Bad" if prediction == 1 else "Good")
st.write(f"**Probability of Bad Risk:** {prediction_proba*100:.2f}%")


st.subheader("Feature Importance")
feature_names = list(encoder.get_feature_names_out(cat_cols)) + list(X.drop(cat_cols, axis=1).columns)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)[:10]
fig, ax = plt.subplots()
feat_imp.plot(kind='barh', ax=ax)
plt.xlabel("Importance")
st.pyplot(fig)

st.markdown("""
---
**Note:**  
- This credit rating indicaters (bad risk = credit amount above median).  
- Made By Shashank Shivam
""")
