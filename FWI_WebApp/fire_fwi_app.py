# FWI Fire Predictor Web App
# Necessary imports

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv('FWI_WebApp/Algerian_Forest_Fire_Cleaned_Dataset.csv')


data = load_data()
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
target = 'FWI'

# Train/test split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Standardization of features
from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X_train_scaled = sts.fit_transform(X_train)
X_test_scaled = sts.transform(X_test)

# Sidebar: Choose model
st.sidebar.title("Choose Regression Model")
model_name = st.sidebar.selectbox("Model", ["Linear", "Lasso", "Ridge", "ElasticNet"])

# Input features
st.sidebar.title("Enter the Features")
user_input = [st.sidebar.number_input(f, value=float(X[f].mean())) for f in features]

# Model function
def get_model(name):
    if name == "Linear":
        return LinearRegression()
    if name == "Lasso":
        return Lasso(alpha=2.0)
    if name == "Ridge":
        return Ridge(alpha=2.0)
    return ElasticNet(alpha=0.1, l1_ratio=0.5)

# Train model
model = get_model(model_name)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Evaluation
st.title("🔥 Algerian Forest FWI (Fire Weather Index) Predictor")
st.subheader(f"{model_name} Regression")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")
st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")

# Predict from user input
if st.sidebar.button("Predict FWI"):
    new_data = np.array(user_input).reshape(1, -1)
    prediction = model.predict(new_data)[0]
    st.sidebar.success(f"Predicted FWI: {prediction:.2f}")

# Visualizations 
st.subheader("Actual vs Predicted")
fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.legend()
ax1.set_xlabel("Actual FWI")
ax1.set_ylabel("Predicted FWI")
st.pyplot(fig1)

st.subheader("Residuals Distribution")
residuals = y_test - y_pred
fig2, ax2 = plt.subplots()
plt.legend()
sns.histplot(residuals, kde=True, ax=ax2, color='orange')
st.pyplot(fig2)
