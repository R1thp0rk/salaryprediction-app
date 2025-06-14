import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the pre-trained model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    try:
        # Try to load the model
        model = joblib.load('salary_predictor.pkl')
        
        # Try to load preprocessor if it exists
        try:
            preprocessor = joblib.load('preprocessor.pkl')
        except FileNotFoundError:
            # If preprocessor doesn't exist, we'll create one
            # You'll need to recreate this based on your training data
            preprocessor = None
            
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Get model feature names to understand what the model expects
def get_model_feature_names():
    """Try to extract feature names from the model"""
    try:
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_
        elif hasattr(model, 'n_features_in_'):
            # We know the number of features but not the names
            return None
        else:
            return None
    except:
        return None

# Create a preprocessing function that matches training data format
def preprocess_input_data(input_data):
    """
    Preprocess input data to match the model's expected format
    """
    try:
        # First, let's see what the model expects
        feature_names = get_model_feature_names()
        
        # if feature_names is not None:
        #     st.info(f"Model expects {len(feature_names)} features")
        #     with st.expander("Expected Features"):
        #         st.write(list(feature_names))
        
        # Strategy 1: Try to create the exact features the model expects
        # We'll manually create dummy variables and ensure they match
        
        processed_data = pd.DataFrame()
        
        # Handle numerical features
        numerical_cols = ['Year of Exp.', 'Age']
        for col in numerical_cols:
            if col in input_data.columns:
                processed_data[col] = input_data[col]
        
        # Handle categorical features - we need to create specific dummy columns
        # Based on common training practices, let's create the likely feature names
        
        categorical_mappings = {
            'Term': ['Full-time', 'Part-time', 'Contract', 'Temporary', 'Internship'],
            'Hiring': ['Direct Hire', 'Recruiter', 'Agency', 'Other'],
            'Industry': ['Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Education', 'Retail', 'Other'],
            'Qualification': ['High School', "Bachelor's", "Master's", 'PhD', 'Professional Certification', 'Other'],
            'Sex': ['Male', 'Female', 'Other'],
            'Language': ['English', 'Spanish', 'French', 'German', 'Other'],
            'Location': ['Urban', 'Suburban', 'Rural', 'Metropolitan', 'Other'],
        }
        
        # Create dummy variables for each categorical column
        for col, categories in categorical_mappings.items():
            if col in input_data.columns:
                for category in categories:
                    # Create column name as it would appear after pd.get_dummies
                    column_name = f"{col}_{category}"
                    processed_data[column_name] = (input_data[col] == category).astype(int)
        
        # Handle text fields that might need special processing
        text_fields = ['Standardized_Job_Title', 'Level_Updated', 'Standardized_Category', 'Standardized_Industry']
        for col in text_fields:
            if col in input_data.columns:
                # For now, let's try to include them as-is or create simple mappings
                if col == 'Level_Updated':
                    levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
                    for level in levels:
                        column_name = f"{col}_{level}"
                        processed_data[column_name] = (input_data[col] == level).astype(int)
                elif col == 'Standardized_Category':
                    categories = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations']
                    for category in categories:
                        column_name = f"{col}_{category}"
                        processed_data[column_name] = (input_data[col] == category).astype(int)
                else:
                    # For job titles and industries, we might need a different approach
                    # For now, let's create a simple encoding
                    processed_data[col] = input_data[col].astype('category').cat.codes
        
        return processed_data
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

model, preprocessor = load_model_and_preprocessor()

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
    
    if model is None:
        st.error("Model could not be loaded. Please check your model file.")
        return
    
    st.write("Please enter your details to get a salary prediction:")
    
    # Input fields
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            term = st.selectbox("Term", ["Full-time", "Part-time"])
            years_exp = st.number_input("Years of Experience", min_value=0, max_value=15, value=5)
            hiring = st.selectbox("Hiring", ["Direct Hire", "Recruiter", "Agency", "Other"])
            industry = st.selectbox("Industry", [
                "Information Technology", "General Business Services", "Human Resource", "Others", "Sales",
                "Automotive", "Education", "Construction", "Financial Services", "Accounting/Audit/Tax Services",
                "Manufacturing", "NGO/Charity/Social Services", "Real Estate", "Exec. / Management",
                "Food & Beverages", "Telecommunication", "Logistics", "Healthcare", "Retail",
                "Hotel/Hospitality", "Trading", "Engineering", "Advertising/Media/Publishing/Printing",
                "Legal Services", "Energy/Power/Water/Oil & Gas", "Customer Service", "Garment Manufacturing",
                "Agriculture", "Research", "Tourism", "Mining", "Entertainment"
            ])

            qualification = st.selectbox("Qualification", [
                "Bachelor Degree", "No limitations", "Associate Degree", "High School", "Master Degree"
            ])

            
        with col2:
            sex = st.selectbox("Sex", ["Male", "Female", "Both"])
            language = st.selectbox("Language", [
                "English", "Chinese", "No need", "Chinese, English", "Japanese, English",
                "Thai, English", "Korean, English", "Vietnamese, English", "French"
            ])
            age = st.selectbox("Age", ["Age Limited", "Unlimited"])
            location = st.selectbox("Location", [
                "Phnom Penh", "Banteay Meanchey", "Siem Reap", "Preah Sihanouk", "Pailin", 
                "Kandal", "Battambang", "Kampong Chhnang", "Kratie", "Pursat", "Takeo", 
                "Mondulkiri", "Kampong Cham", "Tboung Khmum", "Kampong Thom", "Kampot", 
                "Kampong Speu", "Svay Rieng", "Kompong Chhnang", "Koh Kong", "Rattanakiri"
            ])

            job_title = st.text_input("Job Title", "Software Engineer")
            
        # Additional fields that might be required by the model
        col3, col4 = st.columns(2)
        with col3:
            level = st.selectbox("Level", ["Junior", "Mid", "Senior", "Lead", "Manager"])
        with col4:
            category = st.selectbox("Job Category", [
                "Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"
            ])
        
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
                "Level_Updated": [level],  # Now user-selectable
                "Standardized_Category": [category],  # Now user-selectable
                "Standardized_Industry": [industry]  # Same as Industry
            })
            
            try:
                # Check if we have a preprocessor
                if preprocessor is not None:
                    # Use the saved preprocessor
                    processed_data = preprocessor.transform(input_data)
                    st.success("Using saved preprocessor")
                else:
                    # If no preprocessor was saved, use our custom preprocessing
                    #st.warning("No preprocessor found. Using custom preprocessing approach.")
                    processed_data = preprocess_input_data(input_data)
                    
                    if processed_data is None:
                        st.error("Failed to preprocess data")
                        return
                
                # Show preprocessing results
                # with st.expander("Preprocessing Results"):
                #     st.write(f"Processed data shape: {processed_data.shape}")
                #     st.write("Processed features:", processed_data.columns.tolist() if hasattr(processed_data, 'columns') else 'Array format')
                #     if hasattr(processed_data, 'head'):
                #         st.write("Sample processed data:", processed_data.head())
                
                # Try different approaches if the first one fails
                prediction = None
                
                # Approach 1: Direct prediction
                try:
                    prediction = model.predict(processed_data)
                    st.success("✅ Prediction successful with custom preprocessing")
                except Exception as e1:
                    #st.warning(f"First approach failed: {e1}")
                    
                    # Approach 2: Try with just numerical features
                    try:
                        numerical_data = input_data[['Year of Exp.', 'Age']].values
                        #st.info("Trying with only numerical features...")
                        prediction = model.predict(numerical_data.reshape(1, -1))
                        #st.warning("⚠️ Prediction made with only numerical features (less accurate)")
                    except Exception as e2:
                        #st.warning(f"Numerical-only approach failed: {e2}")
                        
                        # Approach 3: Try label encoding
                        try:
                            #st.info("Trying with label encoding...")
                            from sklearn.preprocessing import LabelEncoder
                            
                            input_encoded = input_data.copy()
                            for col in input_data.select_dtypes(include=['object']).columns:
                                le = LabelEncoder()
                                input_encoded[col] = le.fit_transform(input_data[col])
                            
                            prediction = model.predict(input_encoded)
                            
                        except Exception as e3:
                            st.error(f"All approaches failed. Final error: {e3}")
                
                # Display prediction if successful
                if prediction is not None:
                    # Format the prediction nicely
                    if hasattr(prediction, '__len__') and len(prediction) > 0:
                        salary_prediction = prediction[0]
                    else:
                        salary_prediction = prediction
                    
                    st.success(f"🎯 **Predicted Salary: ${salary_prediction:,.2f}**")
                    
                    # Add some context
                    st.info(f"""
                    **Prediction Details:**
                    - Experience: {years_exp} years
                    - Industry: {industry}
                    - Qualification: {qualification}
                    - Employment Type: {term}
                    - Age: {age}
                    """)
                    
                    # Add disclaimer
                    st.caption("⚠️ This prediction is based on the available model. For best accuracy, ensure the model was trained with similar data preprocessing.")
                
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                
                # Enhanced debug information
                with st.expander("🔍 Debug Information"):
                    st.write("**Input data:**")
                    st.write("- Shape:", input_data.shape)
                    st.write("- Columns:", input_data.columns.tolist())
                    st.write("- Data types:", input_data.dtypes.to_dict())
                    st.write("- Sample data:")
                    st.dataframe(input_data)
                    
                    # Model information
                    st.write("**Model information:**")
                    st.write("- Model type:", type(model).__name__)
                    if hasattr(model, 'n_features_in_'):
                        st.write("- Expected features:", model.n_features_in_)
                    if hasattr(model, 'feature_names_in_'):
                        st.write("- Feature names:", model.feature_names_in_)
                    
                    st.write("**Suggested fixes:**")
                    st.write("1. Retrain the model and save both model and preprocessor")
                    st.write("2. Use a Pipeline that includes preprocessing")
                    st.write("3. Ensure feature names match exactly between training and prediction")

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