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
                    st.warning("No preprocessor found. Using custom preprocessing approach.")
                    processed_data = preprocess_input_data(input_data)
                    
                    if processed_data is None:
                        st.error("Failed to preprocess data")
                        return
                
                # Show preprocessing results
                with st.expander("Preprocessing Results"):
                    st.write(f"Processed data shape: {processed_data.shape}")
                    st.write("Processed features:", processed_data.columns.tolist() if hasattr(processed_data, 'columns') else 'Array format')
                    if hasattr(processed_data, 'head'):
                        st.write("Sample processed data:", processed_data.head())
                
                # Try different approaches if the first one fails
                prediction = None
                
                # Approach 1: Direct prediction
                try:
                    prediction = model.predict(processed_data)
                    st.success("✅ Prediction successful with custom preprocessing")
                except Exception as e1:
                    st.warning(f"First approach failed: {e1}")
                    
                    # Approach 2: Try with just numerical features
                    try:
                        numerical_data = input_data[['Year of Exp.', 'Age']].values
                        st.info("Trying with only numerical features...")
                        prediction = model.predict(numerical_data.reshape(1, -1))
                        st.warning("⚠️ Prediction made with only numerical features (less accurate)")
                    except Exception as e2:
                        st.warning(f"Numerical-only approach failed: {e2}")
                        
                        # Approach 3: Try label encoding
                        try:
                            st.info("Trying with label encoding...")
                            from sklearn.preprocessing import LabelEncoder
                            
                            input_encoded = input_data.copy()
                            for col in input_data.select_dtypes(include=['object']).columns:
                                le = LabelEncoder()
                                input_encoded[col] = le.fit_transform(input_data[col])
                            
                            prediction = model.predict(input_encoded)
                            st.warning("⚠️ Prediction made with label encoding (may be inaccurate)")
                        except Exception as e3:
                            st.error(f"All approaches failed. Final error: {e3}")
                
                # Display prediction if successful
                if prediction is not None:
                    # Format the prediction nicely
                    if hasattr(prediction, 'len') and len(prediction) > 0:
                        salary_prediction = prediction[0]
                    else:
                        salary_prediction = prediction
                    
                    st.success(f"🎯 Predicted Salary: ${salary_prediction:,.2f}")
                    
                    # Add some context
                    st.info(f"""
                    Prediction Details:
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
                    st.write("Input data:")
                    st.write("- Shape:", input_data.shape)
                    st.write("- Columns:", input_data.columns.tolist())
                    st.write("- Data types:", input_data.dtypes.to_dict())
                    st.write("- Sample data:")
                    st.dataframe(input_data)
                    
                    # Model information
                    st.write("Model information:")
                    st.write("- Model type:", type(model).name)
                    if hasattr(model, 'n_features_in_'):
                        st.write("- Expected features:", model.n_features_in_)
                    if hasattr(model, 'feature_names_in_'):
                        st.write("- Feature names:", model.feature_names_in_)
                    
                    st.write("Suggested fixes:")
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

if name == "main":
    main()