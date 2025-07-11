import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💼 Salary Prediction using Ensemble Learning</h1>", unsafe_allow_html=True)

# Upload or default data
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Position_Salaries.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df)

# Train-Test Split
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sidebar Model Choice
st.sidebar.title("🔧 Model Settings")
model_type = st.sidebar.radio("Choose Model", ["Random Forest", "Gradient Boosting", "AdaBoost"])

if model_type == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
elif model_type == "Gradient Boosting":
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
else:
    model = AdaBoostRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

st.sidebar.markdown(f"📉 **MSE:** `{mse:.2f}`")

# Prediction Input
level = st.sidebar.slider("Select Position Level", 1.0, 10.0, 5.0, 0.1)
salary_pred = model.predict([[level]])

st.sidebar.success(f"💰 Predicted Salary: ₹{salary_pred[0]:,.2f}")

# Plot
st.subheader("📈 Salary vs Level")
fig, ax = plt.subplots()
sns.scatterplot(x=df["Level"], y=df["Salary"], color='blue', label="Actual", ax=ax)
sns.lineplot(x=df["Level"], y=model.predict(df[["Level"]]), color='red', label="Model Prediction", ax=ax)
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
st.pyplot(fig)

st.info("👈 Use the sidebar to try different models and predict salary based on level.")
