import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score,mean_absolute_error, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime

html_temp = """
    <div style="background-color:#f63366;padding:10px">
    <h2 style="color:white;text-align:center;">Stock Market Prediction App</h2>
    <p style="color:white;text-align:center;" >This app is used for prediction of <b> TESLA </b> Stock price.</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

# Now we are making a select box for dataset
user_input=st.sidebar.text_input('Stock Ticker (based on Yahoo Finance Website)','TSLA')

# The Next is selecting algorithm
# We will display this in the sidebar
algorithm=st.sidebar.selectbox("Select Supervised Learning Algorithm",
                     ("Random Forest Classifier","Linear Regression","Stacking Classifier"))

start = '2010-01-01'
end = datetime.now()
df=yfin.download(user_input,start,end)

st.write(f'## Algorithm is : {algorithm}')

#Subtitle
st.subheader('Data from 2010-Today')

#Data Description
st.write(df.sort_index(ascending=False))

#Chart for Closing Price
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

# Ensure we know the actual closing price
data = df[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = df.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]

# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
tsla_prev = df.copy()
tsla_prev = tsla_prev.shift(1)

# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(tsla_prev[predictors]).iloc[1:]

base_models = [
    ("random_forest", RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)),
    ("linear_regression", LinearRegression())
]

if algorithm == "Random Forest Classifier":
    algo_model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
elif algorithm == "Linear Regression":
    algo_model = LinearRegression()
else:
    algo_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1))

# Now splitting into Testing and Training data
# It will split into 80 % training data and 20 % Testing data
train, test = train_test_split(data, test_size=0.3)
train = data.iloc[:-100]
test = data.iloc[-100:]

# Training algorithm
algo_model.fit(train[predictors], train["Target"])

# Now we will find the predicted values
predict=algo_model.predict(test[predictors])

st.write(f'## Output Results for : {algorithm}')
# Finding Accuracy
# Evaluating/Testing the model
if algorithm != 'Linear Regression':
    # For all algorithm we will find accuracy
    st.write(f"### Training Accuracy is:",algo_model.score(train[predictors],train["Target"])*100)
    st.write(f"### Testing Accuracy is:",accuracy_score(test["Target"],predict)*100)
    st.write(f"### Precision Score::",precision_score(test["Target"],predict))
    fig=plt.figure(figsize=(12,6))
    plt.plot(pd.DataFrame({"Target": test["Target"].values, "Predictions": predict}))
    st.pyplot(fig)
else:
    # Checking for Error
    # Error is less as accuracy is more
    # For linear regression we will find error
    linear_preds_b = (predict > 0.5).astype(int)
    st.write(f"### Mean Squared error is:",mean_squared_error(test["Target"],predict))
    st.write(f"### Mean Absolute error is:",mean_absolute_error(test["Target"],predict))
    st.write(f"### Testing Accuracy is:",accuracy_score(test["Target"],linear_preds_b)*100)
    st.write(f"### Precision Score::",precision_score(test["Target"],linear_preds_b))
    fig=plt.figure(figsize=(12,6))
    plt.plot(pd.DataFrame({"Target": test["Target"].values, "Predictions": linear_preds_b}))
    st.pyplot(fig)
