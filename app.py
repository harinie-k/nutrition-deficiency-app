import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fuzzywuzzy import process
from datetime import date, timedelta
import random

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
# Nutrient calculation with allergy filtering
# ------------------------------
def calculate_nutrients_from_csv(food_list):
    total = {'Iron': 0, 'B12': 0, 'VitD': 0, 'Calcium': 0}
    allergies = []
    if "user_profile" in st.session_state:
        allergies = st.session_state["user_profile"].get("allergies", [])

    for food in food_list:
        food_clean = food.strip().lower()
        if food_clean in allergies:
            st.warning(f"⚠️ Skipped {food} (listed in allergies)")
            continue

        matched = match_food(food)
        if matched:
            row = df_nutrients[df_nutrients['Food_Item'] == matched].iloc[0]
            total['Iron'] += row['Iron_mg']
            total['B12'] += row['B12_ug']
            total['VitD'] += row['VitaminD_IU']
            total['Calcium'] += row['Calcium_mg']
    return total

# =============================
# Default 15-day diet profile
# =============================
default_meals = {
    'Breakfast': ['Idli', 'Dosa', 'Upma', 'Poha', 'Paratha'],
    'Lunch': ['Rice + Dal + Vegetable', 'Chapati + Sabzi', 'Khichdi', 'Curd Rice', 'Vegetable Pulao'],
    'Dinner': ['Dosa + Chutney', 'Roti + Sabzi', 'Khichdi', 'Light Upma', 'Soup + Bread']
}

def generate_15_day_log():
    today = date.today()
    dates = [today - timedelta(days=14 - i) for i in range(15)]
    data = []
    for d in dates:
        breakfast = random.choice(default_meals['Breakfast'])
        lunch = random.choice(default_meals['Lunch'])
        dinner = random.choice(default_meals['Dinner'])
        data.append({'Date': d.strftime('%d %b %Y'), 'Breakfast': breakfast, 'Lunch': lunch, 'Dinner': dinner})
    return pd.DataFrame(data)

# =============================
# Streamlit UI
# =============================
st.title("🥗 AI Nutritional Deficiency Predictor")

# ----------------------------
# User Profile Form
# ----------------------------
st.header("👤 User Profile")

with st.form("user_profile"):
    name = st.text_input("Enter your name")
    age_profile = st.number_input("Age", min_value=5, max_value=100, value=25)
    gender_profile = st.selectbox("Gender", ["Female", "Male", "Other"])
    allergies_input = st.text_area("List any food allergies (comma separated)")
    health_conditions_input = st.text_area("List any health conditions (optional)")

    submitted = st.form_submit_button("Save Profile")

if submitted:
    st.session_state["user_profile"] = {
        "name": name,
        "age": age_profile,
        "gender": gender_profile,
        "allergies": [a.strip().lower() for a in allergies_input.split(",") if a.strip()],
        "health_conditions": health_conditions_input
    }
    st.success(f"Profile saved for {name}!")

# Display saved profile
if "user_profile" in st.session_state:
    st.subheader("✅ Current Profile")
    st.write(st.session_state["user_profile"])

# ----------------------------
# User Inputs (optional overrides)
# ----------------------------
age = st.number_input("Age (for prediction)", min_value=1, max_value=100, value=st.session_state.get("user_profile", {}).get("age", 25))
gender = st.selectbox("Gender (for prediction)", ["Female", "Male", "Other"], index=["Female","Male","Other"].index(st.session_state.get("user_profile", {}).get("gender", "Female")))
weight = st.number_input("Weight (kg)", 20, 200, 55)
height = st.number_input("Height (cm)", 100, 250, 160)

# ----------------------------
# 15-Day Food Log Table
# ----------------------------
st.subheader("📅 15-Day Food Logging (Default Diet Profile)")
st.write("Modify meals if different from the pre-filled typical diet.")

food_log_df = generate_15_day_log()
edited_df = st.data_editor(food_log_df, num_rows="dynamic", use_container_width=True)

# Convert edited table to dictionary for nutrient calculation
food_log = {}
for _, row in edited_df.iterrows():
    food_log[pd.to_datetime(row['Date'])] = [row['Breakfast'], row['Lunch'], row['Dinner']]

# ----------------------------
# Symptoms
# ----------------------------
st.subheader("⚠️ Symptoms")
fatigue = st.checkbox("Fatigue")
pale_skin = st.checkbox("Pale Skin")
hair_loss = st.checkbox("Hair Loss")
tingling = st.checkbox("Tingling Sensation")
bone_pain = st.checkbox("Bone Pain")
irritability = st.checkbox("Irritability")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Deficiency"):
    bmi = weight / ((height / 100) ** 2)
    gender_code = {'Female': 0, 'Male': 1, 'Other': 2}[gender]
    symptom_values = [int(fatigue), int(pale_skin), int(hair_loss), int(tingling), int(bone_pain), int(irritability)]

    # Process 15-day food log
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
    st.subheader("🧪 Average Daily Nutrient Intake (from 15-day log)")
    nutrient_df = pd.DataFrame(avg_nutrients, index=["Avg Amount"]).T
    st.bar_chart(nutrient_df)

    # RDA comparison
    rda = {'Iron': 18, 'B12': 2.4, 'VitD': 600, 'Calcium': 1000}
    comparison_df = pd.DataFrame({'Consumed': avg_nutrients, 'RDA': rda})
    st.subheader("📊 Nutrient Intake vs Daily Requirement")
    st.bar_chart(comparison_df)

    # Personalized Suggestions
    st.subheader("💡 Personalized Suggestions")
    allergy_list = st.session_state.get("user_profile", {}).get("allergies", [])

    if result == "Iron Deficiency":
        suggestions = ["Ragi", "Spinach", "Drumstick Leaves", "Jaggery", "Dates"]
    elif result == "B12 Deficiency":
        suggestions = ["Milk", "Curd", "Paneer", "Eggs", "Fortified Cereals"]
    elif result == "Calcium Deficiency":
        suggestions = ["Milk", "Curd", "Paneer", "Sesame Seeds", "Leafy Greens"]
    elif result == "Vitamin D Deficiency":
        suggestions = ["Sunlight 15–20 min", "Fortified Foods", "Eggs", "Fish"]
    else:
        suggestions = []

    # Filter by allergies
    safe_suggestions = [s for s in suggestions if s.strip().lower() not in allergy_list]

    if safe_suggestions:
        st.info("✅ Suggested Foods: " + ", ".join(safe_suggestions))
    else:
        st.info("✅ No deficiency detected or all suggestions are restricted due to allergies.")
        if result == "No Deficiency":
            st.balloons()
