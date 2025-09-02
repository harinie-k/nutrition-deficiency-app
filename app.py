import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fuzzywuzzy import process
from datetime import date, timedelta

# =============================
# Load model & nutrient data
# =============================
model = joblib.load('nutrition_model.pkl')
df_nutrients = pd.read_csv("indian_food_nutrients.csv")

# ------------------------------
# Fuzzy match for food items
# ------------------------------
def match_food(food_name):
    choices = df_nutrients['Food_Item'].tolist()
    match, score = process.extractOne(food_name.strip().lower(), choices)
    return match if score >= 80 else None

# ------------------------------
# Nutrient calculation
# ------------------------------
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

# =============================
# Streamlit UI
# =============================
st.title("🥗 AI Nutritional Deficiency Predictor")
st.write("Enter your details, health info, and daily food intake to detect possible deficiencies.")

# User info
age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
height = st.number_input("Height (cm)", 100, 250, 160)
weight = st.number_input("Weight (kg)", 20, 200, 55)

# Health info
st.subheader("🩺 Health Information")
allergies = st.multiselect("Do you have any food allergies?", 
                           ["Milk", "Eggs", "Nuts", "Gluten", "Soy", "Seafood", "Other"])
conditions = st.multiselect("Any health conditions?", 
                            ["Diabetes", "Hypertension", "Kidney Issues", "None"])

# ------------------------------
# Daily Food Log Calendar (30 days)
# ------------------------------
st.subheader("📅 30-Day Food Logging")
st.write("Enter your food intake for each day:")

today = date.today()
dates = [today - timedelta(days=i) for i in range(29, -1, -1)]

food_log = {}
for d in dates:
    with st.expander(f"🍽 {d.strftime('%d %b %Y')}"):
        food_items = st.text_input(f"Food items for {d.strftime('%d %b')}", key=f"food_{d}")
        if food_items:
            food_log[d] = food_items.split(",")

# Symptoms
st.subheader("⚠️ Symptoms")
fatigue = st.checkbox("Fatigue")
pale_skin = st.checkbox("Pale Skin")
hair_loss = st.checkbox("Hair Loss")
tingling = st.checkbox("Tingling Sensation")
bone_pain = st.checkbox("Bone Pain")
irritability = st.checkbox("Irritability")

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Deficiency"):
    bmi = weight / ((height / 100) ** 2)
    gender_code = {'Female': 0, 'Male': 1, 'Other': 2}[gender]
    symptom_values = [int(fatigue), int(pale_skin), int(hair_loss), int(tingling), int(bone_pain), int(irritability)]

    # Process 30-day food log
    total_nutrients = {'Iron': 0, 'B12': 0, 'VitD': 0, 'Calcium': 0}
    for day, foods in food_log.items():
        nutrients = calculate_nutrients_from_csv(foods)
        for k in total_nutrients:
            total_nutrients[k] += nutrients[k]

    days_count = len(food_log) if len(food_log) > 0 else 1
    avg_nutrients = {k: v / days_count for k, v in total_nutrients.items()}

    food_score = int(sum(avg_nutrients.values()))
    input_data = [age, gender_code, weight, height, bmi, food_score] + symptom_values
    input_df = pd.DataFrame([input_data], columns=[
        'Age', 'Gender', 'Weight', 'Height', 'BMMI', 'Food_Log_Label',
        'Fatigue', 'Pale_Skin', 'Hair_Loss', 'Tingling_Sensation',
        'Bone_Pain', 'Irritability'
    ])

    # Model prediction
    prediction = model.predict(input_df)[0]
    label_map = {0: "B12 Deficiency", 1: "Calcium Deficiency", 2: "Iron Deficiency", 3: "No Deficiency", 4: "Vitamin D Deficiency"}
    result = label_map.get(prediction, "Unknown")
    st.success(f"🎯 Predicted Result: **{result}**")

    # Nutrient charts
    st.subheader("🧪 Average Daily Nutrient Intake (from 30-day log)")
    nutrient_df = pd.DataFrame(avg_nutrients, index=["Avg Amount"]).T
    st.bar_chart(nutrient_df)

    # RDA comparison
    rda = {'Iron': 18, 'B12': 2.4, 'VitD': 600, 'Calcium': 1000}
    comparison_df = pd.DataFrame({'Consumed': avg_nutrients, 'RDA': rda})
    st.subheader("📊 Nutrient Intake vs Daily Requirement")
    st.bar_chart(comparison_df)

    # Personalized Suggestions
    st.subheader("💡 Personalized Suggestions")
    if result == "Iron Deficiency":
        if "Milk" in allergies:
            st.info("🩸 Suggestion: Eat spinach, jaggery, drumstick leaves (avoid dairy-based iron sources).")
        else:
            st.info("🩸 Suggestion: Eat ragi, spinach, drumstick leaves, jaggery.")
    elif result == "B12 Deficiency":
        if "Milk" in allergies:
            st.info("🐄 Suggestion: Try eggs, fortified cereals (avoid milk/curd).")
        else:
            st.info("🐄 Suggestion: Add milk, curd, paneer, eggs.")
    elif result == "Calcium Deficiency":
        if "Milk" in allergies:
            st.info("🦴 Suggestion: Include sesame seeds, leafy greens.")
        else:
            st.info("🦴 Suggestion: Include sesame, milk, ragi.")
    elif result == "Vitamin D Deficiency":
        st.info("☀️ Suggestion: Get 15–20 min of sunlight daily, drink fortified foods.")
    else:
        st.balloons()
        st.success("🎉 Great! No deficiency detected.")
