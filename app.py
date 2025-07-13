import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('nutrition_model.pkl')

# Sample food-nutrient mapping
food_nutrient_table = {
    'rice': {'Iron': 0.2, 'B12': 0.0, 'VitD': 0.0, 'Calcium': 10},
    'curd': {'Iron': 0.1, 'B12': 0.4, 'VitD': 5, 'Calcium': 120},
    'spinach': {'Iron': 2.7, 'B12': 0.0, 'VitD': 0.0, 'Calcium': 99},
    'milk': {'Iron': 0.0, 'B12': 1.0, 'VitD': 40, 'Calcium': 125},
    'ragi': {'Iron': 3.9, 'B12': 0.0, 'VitD': 0.0, 'Calcium': 344},
    'paneer': {'Iron': 0.5, 'B12': 1.1, 'VitD': 7, 'Calcium': 208},
}

# Function to calculate nutrient totals from food list
def calculate_nutrients(food_list):
    total = {'Iron': 0, 'B12': 0, 'VitD': 0, 'Calcium': 0}
    for food in food_list:
        food = food.lower().strip()
        if food in food_nutrient_table:
            for nutrient in total:
                total[nutrient] += food_nutrient_table[food][nutrient]
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
    nutrients = calculate_nutrients(food_list)
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

