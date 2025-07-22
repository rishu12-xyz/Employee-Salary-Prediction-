import streamlit as st
import pandas as pd
import joblib

# Label encoders mapping (must match the order from training)
workclass_map = {'Private': 3, 'Self-emp-not-inc': 5, 'Self-emp-inc': 4, 'Federal-gov': 0, 'Local-gov': 2, 'State-gov': 6, 'Others': 1}
marital_status_map = {'Never-married': 1, 'Married-civ-spouse': 0, 'Divorced': 2, 'Separated': 2, 'Widowed': 2, 'Married-spouse-absent': 2}
occupation_map = {'Tech-support': 11, 'Craft-repair': 0, 'Other-service': 7, 'Sales': 9, 'Exec-managerial': 3, 'Prof-specialty': 8, 'Handlers-cleaners': 4, 'Machine-op-inspct': 6, 'Adm-clerical': 1, 'Farming-fishing': 2, 'Transport-moving': 10, 'Priv-house-serv': 5, 'Protective-serv': 12, 'Armed-Forces': 13, 'Others': 7}
relationship_map = {'Wife': 4, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 0, 'Unmarried': 3}
race_map = {'White': 4, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 0, 'Other': 2, 'Black': 3}
gender_map = {'Male': 1, 'Female': 0}
native_country_map = {'United-States': 38, 'Others': 0}  # Simplified for demo

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Input fields matching model features
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
native_country = st.sidebar.selectbox("Native Country", list(native_country_map.keys()))
educational_num = st.sidebar.slider("Education Number", 5, 16, 10)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Preprocess input
input_dict = {
    'age': age,
    'workclass': workclass_map[workclass],
    'marital-status': marital_status_map[marital_status],
    'occupation': occupation_map[occupation],
    'relationship': relationship_map[relationship],
    'race': race_map[race],
    'gender': gender_map[gender],
    'native-country': native_country_map[native_country],
    'educational-num': educational_num,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week
}
input_df = pd.DataFrame([input_dict])

st.write("### 🔎 Input Data (preprocessed)")
st.write(input_df)

if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    # Preprocess batch data (must match above, user must provide correct columns)
    for col, mapping in zip(
        ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'],
        [workclass_map, marital_status_map, occupation_map, relationship_map, race_map, gender_map, native_country_map]
    ):
        if col in batch_data.columns:
            batch_data[col] = batch_data[col].map(mapping).fillna(0).astype(int)

    # Ensure all required columns are present and in the correct order
    feature_order = [
        'age',
        'workclass',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'gender',
        'native-country',
        'educational-num',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]
    # Fill missing columns with 0 or a safe default
    for col in feature_order:
        if col not in batch_data.columns:
            batch_data[col] = 0
    batch_data = batch_data[feature_order]

    st.write("Uploaded data preview (preprocessed):", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

