import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_predict_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Acceptance Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package.
Enter customer data below to get a prediction.
""")

# Input fields for tourism dataset features
age = st.number_input("Age", 18, 99, 30)
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 0.0, 60.0, 10.0, 0.5)
occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business', 'Government Sector', 'Student'])
gender = st.selectbox("Gender", ['Female', 'Male'])
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 10, 2)
number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
marital_status = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married'])
number_of_trips = st.number_input("Number of Trips Annually", 0, 50, 5)
passport = st.selectbox("Has Passport?", ['Yes', 'No'])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Owns Car?", ['Yes', 'No'])
number_of_children_visiting = st.number_input("Number of Children Visiting (below 5)", 0, 5, 0)
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("Monthly Income", 0.0, 100000.0, 25000.0, 100.0)

# Convert 'Yes'/'No' to 1/0 for Passport and OwnCar
passport_val = 1 if passport == 'Yes' else 0
own_car_val = 1 if own_car == 'Yes' else 0

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport_val,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car_val,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict Acceptance"):
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    # Using the classification_threshold from the training script
    classification_threshold = 0.45
    prediction = 1 if prediction_proba >= classification_threshold else 0

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"The model predicts: **Customer WILL likely purchase the package!** (Probability: {prediction_proba:.2f})")
    else:
        st.warning(f"The model predicts: **Customer will NOT likely purchase the package.** (Probability: {prediction_proba:.2f})")
