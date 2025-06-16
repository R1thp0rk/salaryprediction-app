import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Cambodia Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# User credentials (in production, use proper authentication)
VALID_CREDENTIALS = {
    "admin": "admin123",
    "hr_manager": "hr2024",
    "recruiter": "recruit123",
    "demo_user": "demo123"
}

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the optimized model
        model = joblib.load('cambodia_salary_predictor.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'cambodia_salary_predictor.pkl' not found. Please train and save the model first.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Preprocessing function to match training data
def preprocess_user_input(user_data, numerical_cols, categorical_cols):
    """
    Preprocess user input to match the model's expected format
    """
    try:
        # Create preprocessing pipeline matching the training setup
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Fit and transform the input data
        # Note: In production, you should save and load the fitted preprocessor
        # For now, we'll create a minimal fit on the input data
        processed_data = preprocessor.fit_transform(user_data)
        
        return processed_data
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None

# Login function
def login():
    st.markdown('<h1 class="main-header">🔐 Cambodia Salary Predictor</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please login to continue")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if submit_button:
                if username in VALID_CREDENTIALS and password == VALID_CREDENTIALS[username]:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # Demo credentials info
        with st.expander("🔍 Demo Credentials"):
            st.write("**Available demo accounts:**")
            st.code("""
Username: demo_user
Password: demo123

Username: admin  
Password: admin123
            """)

# Main salary prediction app
def salary_predictor():
    # Header
    st.markdown('<h1 class="main-header">💰 Cambodia Salary Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.username}!")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 About This App")
        st.info("""
        This app predicts salaries in Cambodia's job market based on:
        - Experience & Age
        - Industry & Location  
        - Education & Skills
        - Employment Type
        """)
        
        st.markdown("### 🎯 Model Performance")
        st.metric("R² Score", "0.42", "Explains 42% of variance")
        st.metric("Accuracy", "90%", "For salary range classification")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Main prediction interface
    st.markdown("### 📝 Enter Your Details")
    
    with st.form("prediction_form", clear_on_submit=False):
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            age = st.number_input("Age", min_value=18, max_value=65, value=28, help="Your current age")
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=40, value=3, help="Total years of work experience")
            sex = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
            
            st.markdown("#### 🎓 Education & Skills")
            qualification = st.selectbox("Highest Qualification", [
                "High School", "Associate Degree", "Bachelor Degree", 
                "Master Degree", "PhD", "Professional Certification"
            ], index=2, help="Your highest educational qualification")
            
            language = st.selectbox("Language Skills", [
                "English", "Chinese", "Chinese, English", "Japanese, English",
                "Thai, English", "Korean, English", "Vietnamese, English", 
                "French", "No specific requirement"
            ], help="Primary language skills required")
        
        with col2:
            st.markdown("#### 💼 Job Information")
            term = st.selectbox("Employment Type", [
                "Full-time", "Part-time", "Contract", "Temporary", "Internship"
            ], help="Type of employment")
            
            industry = st.selectbox("Industry", [
                "Information Technology", "Financial Services", "Healthcare", 
                "Education", "Manufacturing", "Retail", "Construction",
                "Real Estate", "Hospitality", "Automotive", "Agriculture",
                "Energy", "Telecommunications", "Media", "Legal Services",
                "Consulting", "NGO/Non-profit", "Government", "Other"
            ], help="Industry sector")
            
            hiring = st.selectbox("Hiring Channel", [
                "Direct Hire", "Recruiter", "Agency", "Internal Transfer", "Other"
            ], help="How you were hired")
            
            st.markdown("#### 📍 Location")
            location = st.selectbox("Work Location", [
                "Phnom Penh", "Siem Reap", "Battambang", "Preah Sihanouk", 
                "Kandal", "Kampong Cham", "Kampot", "Takeo", "Pursat",
                "Banteay Meanchey", "Svay Rieng", "Kampong Chhnang",
                "Kampong Speu", "Kampong Thom", "Kratie", "Mondulkiri",
                "Pailin", "Koh Kong", "Rattanakiri", "Stung Treng"
            ], help="Province where you work")
        
        # Additional fields
        col3, col4 = st.columns(2)
        with col3:
            job_level = st.selectbox("Job Level", [
                "Entry Level", "Junior", "Mid-Level", "Senior", "Lead", 
                "Manager", "Director", "Executive"
            ], index=2, help="Your current job level")
        
        with col4:
            job_category = st.selectbox("Job Category", [
                "Engineering", "Sales & Marketing", "Finance & Accounting", 
                "Human Resources", "Operations", "Customer Service",
                "Research & Development", "Quality Assurance", "Legal",
                "Administration", "Creative", "Other"
            ], help="Primary job function category")
        
        # Submit button
        st.markdown("---")
        col_submit1, col_submit2, col_submit3 = st.columns([1, 1, 1])
        with col_submit2:
            submit_button = st.form_submit_button("🎯 Predict My Salary", use_container_width=True)
    
    # Make prediction when form is submitted
    if submit_button:
        try:
            # Prepare input data
            input_data = pd.DataFrame({
                'Age': [age],
                'Year of Exp.': [years_exp],
                'Sex': [sex],
                'Qualification': [qualification],
                'Language': [language],
                'Term': [term],
                'Industry': [industry],
                'Hiring': [hiring],
                'Location': [location],
                'Job_Level': [job_level],
                'Job_Category': [job_category]
            })
            
            # Display input summary
            with st.expander("📋 Input Summary", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Personal Details:**")
                    st.write(f"• Age: {age} years")
                    st.write(f"• Experience: {years_exp} years")
                    st.write(f"• Gender: {sex}")
                    st.write(f"• Education: {qualification}")
                
                with col_b:
                    st.write("**Job Details:**")
                    st.write(f"• Industry: {industry}")
                    st.write(f"• Location: {location}")
                    st.write(f"• Employment: {term}")
                    st.write(f"• Level: {job_level}")
            
            # Make prediction using the model
            # Note: This assumes the model can handle the input directly
            # In practice, you might need to preprocess the data to match training format
            
            with st.spinner("🔮 Calculating your salary prediction..."):
                try:
                    # Simple approach - try direct prediction
                    # You may need to adjust this based on how your model was trained
                    prediction = model.predict(input_data)
                    
                    if hasattr(prediction, '__len__') and len(prediction) > 0:
                        salary_prediction = prediction[0]
                    else:
                        salary_prediction = prediction
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎉 Your Predicted Salary</h2>
                        <h1>${salary_prediction:,.2f} USD</h1>
                        <p>Based on Cambodia job market data</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional insights
                    col_insight1, col_insight2, col_insight3 = st.columns(3)
                    
                    with col_insight1:
                        monthly_salary = salary_prediction / 12
                        st.metric("💰 Monthly Salary", f"${monthly_salary:,.2f}")
                    
                    with col_insight2:
                        # Calculate salary range (±15%)
                        salary_range = salary_prediction * 0.15
                        st.metric("📊 Salary Range", f"±${salary_range:,.2f}")
                    
                    with col_insight3:
                        # Experience factor
                        if years_exp < 2:
                            exp_level = "Entry Level"
                        elif years_exp < 5:
                            exp_level = "Junior"
                        elif years_exp < 10:
                            exp_level = "Mid-Level"
                        else:
                            exp_level = "Senior"
                        st.metric("📈 Experience Level", exp_level)
                    
                    # Salary insights
                    st.markdown("### 💡 Salary Insights")
                    
                    col_tips1, col_tips2 = st.columns(2)
                    
                    with col_tips1:
                        st.markdown("""
                        <div class="info-box">
                            <h4>🚀 Ways to Increase Salary:</h4>
                            <ul>
                                <li>Gain more experience in your field</li>
                                <li>Pursue higher education or certifications</li>
                                <li>Develop language skills (especially English)</li>
                                <li>Consider high-demand industries like IT</li>
                                <li>Build leadership and management skills</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_tips2:
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>📍 Market Context:</h4>
                            <ul>
                                <li>Location: {location} - {"High" if location == "Phnom Penh" else "Moderate"} salary market</li>
                                <li>Industry: {industry} - {"Growing" if industry in ["Information Technology", "Financial Services"] else "Stable"} sector</li>
                                <li>Experience: {years_exp} years - {"Junior" if years_exp < 3 else "Experienced"} level</li>
                                <li>Education: {qualification} - {"Advanced" if "Master" in qualification or "PhD" in qualification else "Standard"} qualification</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Disclaimer
                    st.markdown("---")
                    st.caption("""
                    ⚠️ **Disclaimer:** This prediction is based on historical data and machine learning models. 
                    Actual salaries may vary based on company size, specific skills, market conditions, and other factors. 
                    Use this as a general guideline for salary expectations in Cambodia's job market.
                    """)
                    
                except Exception as prediction_error:
                    st.error(f"❌ Prediction failed: {str(prediction_error)}")
                    
                    # Provide fallback estimation
                    st.warning("Using simplified estimation...")
                    
                    # Simple salary estimation based on basic factors
                    base_salary = 8000  # Base salary in USD
                    exp_multiplier = 1 + (years_exp * 0.1)  # 10% increase per year
                    education_bonus = {"High School": 1.0, "Associate Degree": 1.1, 
                                     "Bachelor Degree": 1.3, "Master Degree": 1.6, "PhD": 2.0}
                    location_bonus = 1.3 if location == "Phnom Penh" else 1.0
                    
                    estimated_salary = (base_salary * exp_multiplier * 
                                      education_bonus.get(qualification, 1.2) * location_bonus)
                    
                    st.info(f"📊 **Estimated Salary:** ${estimated_salary:,.2f} (Simplified calculation)")
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")

# Main application flow
def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    # Route to appropriate page
    if not st.session_state.logged_in:
        login()
    else:
        salary_predictor()

# Run the app
if __name__ == "__main__":
    main()