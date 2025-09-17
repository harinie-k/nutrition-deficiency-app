import streamlit as st
import pandas as pd
import numpy as np
import joblib
from fuzzywuzzy import process
from datetime import date, timedelta
import random
import json
import os

# =============================
# AUTHENTICATION (Simple JSON-based)
# =============================
USER_DB = "users.json"

def load_users():
    if not os.path.exists(USER_DB):
        return {}
    with open(USER_DB, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f)

def signup(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {"password": password}
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    if username in users and users[username]["password"] == password:
        return True
    return False

# =============================
# NUTRITION MODEL & DATA
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
# MAIN APP FUNCTION
# =============================
def main_app():
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
    # 15-Day Food Log Table
    # ------------------------------
    st.subheader("📅 15-Day Food Logging (Default Diet Profile)")
    st.write("Modify meals if different from the pre-filled typical diet.")

    food_log_df = generate_15_day_log()
    edited_df = st.data_editor(food_log_df, num_rows="dynamic", use_container_width=True)

    food_log = {}
    for _, row in edited_df.iterrows():
        food_log[pd.to_datetime(row['Date'])] = [row['Breakfast'], row['Lunch'], row['Dinner']]

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

# =============================
# LOGIN / SIGNUP INTERFACE
# =============================
def login_signup_page():
    st.title("🔐 User Authentication")

    option = st.radio("Choose Action", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if signup(username, password):
                st.success("✅ Account created! Please login now.")
            else:
                st.error("⚠️ Username already exists!")
    else:  # Login
        if st.button("Login"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("❌ Invalid credentials")

# =============================
# APP ENTRY POINT
# =============================
if "logged_in" not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    login_signup_page()
