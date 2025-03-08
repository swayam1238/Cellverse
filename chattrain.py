import streamlit as st
import pandas as pd
import ollama
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

CSV_PATH = "generated_data.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['patient_id'], errors='ignore')
    df.insert(0, 'patient_num', range(1, len(df) + 1))
    return df

def train_model(df):
    df = df.drop(columns=['patient_num'], errors='ignore')
    df = df.dropna(subset=['recommended_technique'])
    df = df[df['recommended_technique'] != ""]
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        if col != 'recommended_technique':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    features = df.drop(columns=['recommended_technique'])
    target = df['recommended_technique']
    
    target_encoder = LabelEncoder()
    target_encoded = target_encoder.fit_transform(target)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    with open("surgery_model.pkl", "wb") as f:
        pickle.dump((model, target_encoder, label_encoders, features.columns.tolist()), f)

def predict_technique(patient_data):
    with open("surgery_model.pkl", "rb") as f:
        model, target_encoder, label_encoders, feature_names = pickle.load(f)
    
    df = load_data()
    patient_df = pd.DataFrame([patient_data])
    
    if 'recommended_technique' in patient_df:
        patient_df = patient_df.drop(columns=['recommended_technique'])
    
    for col, le in label_encoders.items():
        if col in patient_df.columns:
            patient_df[col] = patient_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([df[col].mode()[0]])[0]
            )
    
    patient_df = patient_df[feature_names]
    prediction = model.predict(patient_df)
    return target_encoder.inverse_transform(prediction)[0]

def generate_feedback(patient_data, additional_prompt):
    predicted_technique = predict_technique(patient_data)
    
    prompt = f"""
    Given the following patient data for cataract surgery, provide:
    - Suggested surgical technique (Predicted: {predicted_technique})
    - Potential risk factors
    - Success rate estimate
    - Personalized recommendations
    - Potential side effects
    - Recovery time
    - Post-operative care instructions
    - Step-by-Step Surgery Guide
    {additional_prompt}
    
    Patient Data:
    {json.dumps(patient_data, indent=2)}
    """
    response = ollama.chat(model='mistral', messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

st.title("AI-Powered Cataract Surgery Assistant")
df = load_data()
train_model(df)

option = st.radio("Select Processing Mode:", ("Select Specific Patient", "Predict for New Patient"))

if option == "Select Specific Patient":
    patient_num = st.selectbox("Choose Patient Number", df['patient_num'])
    patient_data = df[df['patient_num'] == patient_num].drop(columns=['patient_num', 'recommended_technique']).to_dict(orient='records')[0]
    st.write("### Selected Patient Data")
    st.json(patient_data)

elif option == "Predict for New Patient":
    patient_data = {}
    for col in df.columns[1:-1]:
        patient_data[col] = st.text_input(f"Enter {col}")
    
    if st.button("Predict Surgery Technique"):
        prediction = predict_technique(patient_data)
        st.success(f"Predicted Surgical Technique: {prediction}")

additional_prompt = st.text_area("Optional: Add specific feedback request (e.g., 'What should be improved?')")

if st.button("Generate Feedback"):
    with st.spinner("Processing... This may take a moment."):
        feedback = generate_feedback(patient_data, additional_prompt)
        st.write("### AI-Generated Surgical Feedback")
        st.text_area("Feedback", feedback if feedback else "No feedback generated.", height=300)
