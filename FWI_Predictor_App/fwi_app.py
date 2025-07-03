import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 

st.set_page_config(page_title = "🔥 FWI (Fire Weather Index) Predictor", layout = "wide")
st.markdown("<h1 style='text-align: center;'>🔥 Algerian Forest Fire Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict Fire Weather Index (FWI)</h4>", unsafe_allow_html=True)

# Loading the dataset
@st.cache_data
def load_data():
    return pd.read_csv('FWI_Predictor_App/Algerian_Forest_Fire_Cleaned_Dataset.csv')

data = load_data()
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
target = 'FWI'
X, y = data[features], data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


with st.sidebar:
    st.title("⚙️ Model & Input")
    model_name = st.selectbox("Choose a Regression Model", ["Linear", "Lasso", "Ridge", "ElasticNet"])

    st.markdown("### Input Feature Values")
    inputs = [st.number_input(f, value = float(X[f].mean())) for f in features]

def get_model(name):
    if name == "Linear": return LinearRegression()
    if name == "Lasso": return Lasso(alpha=0.1)
    
    if name == "Ridge": return Ridge(alpha=1.0)
    return ElasticNet(alpha=0.1, l1_ratio=0.5)

# Standardization of features
from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X_train_scaled = sts.fit_transform(X_train)
X_test_scaled = sts.transform(X_test)

# Training the model and predicting 
model = get_model(model_name)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.markdown(f"### 📈 {model_name} Regression Results")
col1, col2 = st.columns(2)
col1.metric("R² Score (Accuracy)", f"{r2_score(y_test, y_pred):.2f}")
col2.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")

if st.sidebar.button("Predict FWI"):
    user_data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(user_data)[0]
    st.sidebar.success(f"Predicted FWI: {prediction:.2f}")

tab1, tab2, tab3 = st.tabs(["📍 Actual vs Predicted", "📉 Residuals", "Correlation"])

with tab1:
    st.markdown("#### Predicted vs Actual FWI")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, c='deepskyblue', alpha=0.7)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax1.set_xlabel("Actual FWI")
    ax1.set_ylabel("Predicted FWI")
    ax1.set_title("Predicted vs Actual FWI")
    st.pyplot(fig1)

with tab2:
    st.markdown("#### 📉 Residuals Distribution")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    sns.histplot(residuals, kde=True, color='salmon', ax=ax2, bins=30)
    ax2.set_title("Residuals of Predictions")
    st.pyplot(fig2)

with tab3:
    st.markdown("#### 🧪 Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(data[features + [target]].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

st.markdown("---")
st.markdown("<small style='text-align: center; display: block;'>Copyright</small>", unsafe_allow_html=True)
