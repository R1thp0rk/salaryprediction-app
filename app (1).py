import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open('salary_predictor.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# User credentials (in a real app, use proper authentication)
VALID_CREDENTIALS = {
    "admin": "admin123",
    "user1": "password1",
    "user2": "password2"
}

# Login function
def login():
    st.title("Salary Predictor Login")
    
    # Create login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            if username in VALID_CREDENTIALS and password == VALID_CREDENTIALS[username]:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# Main app function
def salary_predictor():
    st.title("Salary Prediction App")
    st.write(f"Welcome, {st.session_state.username}!")
    
    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.write("Please enter your details to get a salary prediction:")
    
    # Input fields
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            term = st.selectbox("Term", ["Full-time", "Part-time", "Contract", "Temporary", "Internship"])
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
            hiring = st.selectbox("Hiring", ["Direct Hire", "Recruiter", "Agency", "Other"])
            industry = st.selectbox("Industry", [
                "Technology", "Finance", "Healthcare", "Manufacturing", 
                "Education", "Retail", "Other"
            ])
            qualification = st.selectbox("Qualification", [
                "High School", "Bachelor's", "Master's", "PhD", 
                "Professional Certification", "Other"
            ])
            
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
            language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Other"])
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            location = st.selectbox("Location", [
                "Urban", "Suburban", "Rural", "Metropolitan", "Other"
            ])
            job_title = st.text_input("Job Title", "Software Engineer")
        
        submit_button = st.form_submit_button("Predict Salary")
        
        if submit_button:
            # Prepare input data
            input_data = pd.DataFrame({
                "Term": [term],
                "Year of Exp.": [years_exp],
                "Hiring": [hiring],
                "Industry": [industry],
                "Qualification": [qualification],
                "Sex": [sex],
                "Language": [language],
                "Age": [age],
                "Location": [location],
                "Standardized_Job_Title": [job_title],
                # These fields might be needed based on your model
                "Level_Updated": ["Mid"],  # Example value
                "Standardized_Category": ["Engineering"],  # Example value
                "Standardized_Industry": [industry]  # Same as Industry
            })
            
            try:
                # Make prediction
                prediction = model.predict(input_data)
                st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Main app flow
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login()
    else:
        salary_predictor()

if __name__ == "__main__":
    main()