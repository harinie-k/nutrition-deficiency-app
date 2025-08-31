#import streamlit as st

#st.warning("🚧 This app is temporarily under maintenance. Please check back later.")
#st.stop()

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('nutrition_model.pkl')

# Sample food-nutrient mapping
import pandas as pd
from fuzzywuzzy import process

# Load nutrient data from CSV
df_nutrients = pd.read_csv("indian_food_nutrients.csv")

# Fuzzy match function
def match_food(food_name):
    choices = df_nutrients['Food_Item'].tolist()
    match, score = process.extractOne(food_name.strip().lower(), choices)
    return match if score >= 80 else None

# Nutrient calculation using CSV and fuzzy match
def calculate_nutrients_from_csv(food_list):
    total = {'Iron': 0, 'B12': 0, 'VitD': 0, 'Calcium': 0}
    
    for food in food_list:
        matched = match_food(food)
        if matched:
            row = df_nutrients[df_nutrients['Food_Item'] == matched].iloc[0]
            total['Iron'] += row['Iron_mg']
            total['B12'] += row['B12_ug']
            total['VitD'] += row['VitaminD_IU']
            total['Calcium'] += row['Calcium_mg']
    return total


# Streamlit UI
st.title("🥗 AI Nutritional Deficiency Predictor")
st.write("Enter your details and food intake to detect possible deficiencies.")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
height = st.number_input("Height (cm)", 100, 250, 160)
weight = st.number_input("Weight (kg)", 20, 200, 55)
food_input = st.text_input("Food items (comma separated)", "rice, curd, spinach")

st.subheader("Symptoms")
fatigue = st.checkbox("Fatigue")
pale_skin = st.checkbox("Pale Skin")
hair_loss = st.checkbox("Hair Loss")
tingling = st.checkbox("Tingling Sensation")
bone_pain = st.checkbox("Bone Pain")
irritability = st.checkbox("Irritability")

# On Predict button click
if st.button("Predict Deficiency"):
    # Convert inputs
    bmi = weight / ((height / 100) ** 2)
    gender_code = {'Female': 0, 'Male': 1, 'Other': 2}[gender]
    symptoms = [fatigue, pale_skin, hair_loss, tingling, bone_pain, irritability]
    symptom_values = [int(s) for s in symptoms]
    
    # Nutrients from food
    food_list = food_input.split(",")
    nutrients = calculate_nutrients_from_csv(food_list)

    food_score = int(sum(nutrients.values())) 
    
    # Final input vector
    input_data = [
    age, gender_code, weight, height, bmi, food_score
    ] + [int(fatigue), int(pale_skin), int(hair_loss), int(tingling), int(bone_pain), int(irritability)]

# Match the exact features your model was trained on
    input_df = pd.DataFrame([input_data], columns=[
    'Age', 'Gender', 'Weight', 'Height', 'BMMI', 'Food_Log_Label',
    'Fatigue', 'Pale_Skin', 'Hair_Loss', 'Tingling_Sensation',
    'Bone_Pain', 'Irritability'
    ])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Interpret result
    label_map = {0: "B12 Deficiency", 1: "Calcium Deficiency", 2: "Iron Deficiency", 3: "No Deficiency", 4: "Vitamin D Deficiency"}
    result = label_map.get(prediction, "Unknown")

    st.success(f"🎯 Predicted Result: **{result}**")
   
    # Show Nutrient Intake as a Bar Chart
    st.subheader("🧪 Nutrient Intake from Food Log")
    nutrient_df = pd.DataFrame(nutrients, index=["Amount (from input)"]).T
    st.bar_chart(nutrient_df)

    # Recommended Daily Allowance (RDA) values
    rda = {
        'Iron': 18,       # mg
        'B12': 2.4,       # µg
        'VitD': 600,      # IU
        'Calcium': 1000   # mg
    }

    # Prepare comparison DataFrame
    comparison_df = pd.DataFrame({
        'Consumed': nutrients,
        'RDA': rda
    })

    st.subheader("📊 Nutrient Intake vs Daily Requirement")
    st.bar_chart(comparison_df)

    
    # Suggestions
    if result == "Iron Deficiency":
        st.info("🩸 Suggestion: Eat ragi, spinach, drumstick leaves, jaggery.")
    elif result == "B12 Deficiency":
        st.info("🐄 Suggestion: Add milk, curd, paneer, eggs.")
    elif result == "Calcium Deficiency":
        st.info("🦴 Suggestion: Include sesame, milk, ragi.")
    elif result == "Vitamin D Deficiency":
        st.info("☀️ Suggestion: Get 15–20 min of sunlight daily, drink fortified milk.")
    else:
        st.balloons()
        st.success("🎉 Great! No deficiency detected.")


