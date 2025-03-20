import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import datetime

# Initialize session state if not already set
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "session_history" not in st.session_state:
    st.session_state.session_history = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "goals" not in st.session_state:
    st.session_state.goals = {}

# Title of the app
st.title("Fitness Tracker")

# Login Section
st.sidebar.header("User Login")
username = st.sidebar.text_input("Username", key="username")
password = st.sidebar.text_input("Password", type="password", key="password")

if st.sidebar.button("Login"):
    if username and password:
        if username in st.session_state.user_data and st.session_state.user_data[username]["password"] == password:
            st.session_state.logged_in = True
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.session_state.user_data[username] = {"password": password}  # Save new user
            st.session_state.logged_in = True
            st.sidebar.success(f"New user created and logged in as {username}")
    else:
        st.sidebar.error("Please enter a username and password")

# User Profile Section
if st.session_state.logged_in:
    st.sidebar.header("Profile & Goals")
    age = st.sidebar.slider("Age", 10, 80, 25)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    height = st.sidebar.slider("Height (cm)", 100, 220, 170)
    goal = st.sidebar.text_input("Fitness Goal", "Lose Weight")
    st.session_state.user_data[username].update({"age": age, "weight": weight, "height": height, "goal": goal})
    
    # Goal Setting and Tracking
    st.sidebar.subheader("Set Your Goals")
    goal_type = st.sidebar.selectbox("Goal Type", ["Weight Loss", "Muscle Gain", "Endurance", "General Fitness"])
    goal_target = st.sidebar.number_input("Target (kg lost/gained, minutes, etc.)", min_value=0.0, value=5.0)
    st.session_state.goals[username] = {"type": goal_type, "target": goal_target}
    
    st.subheader("User Dashboard")
    st.write(f"*Username:* {username}")
    st.write(f"*Age:* {age}")
    st.write(f"*Weight:* {weight} kg")
    st.write(f"*Height:* {height} cm")
    st.write(f"*Fitness Goal:* {goal}")
    
    if username in st.session_state.goals:
        st.subheader("Your Goal")
        st.write(f"*Goal Type:* {st.session_state.goals[username]['type']}")
        st.write(f"*Target:* {st.session_state.goals[username]['target']}")

    # Load dataset
    data = pd.read_csv("calories.csv")
    X = data[['Age', 'Weight', 'Height', 'Duration', 'Heart_Rate']]
    y = data['Calories']

    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User Inputs
    st.sidebar.header("Enter Workout Details")
    duration = st.sidebar.slider("Exercise Duration (min)", 5, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm)", 50, 180, 100)
    user_input = np.array([[age, weight, height, duration, heart_rate]])
    user_data_df = pd.DataFrame(user_input, columns=X.columns)
    user_data_scaled = scaler.transform(user_data_df)
    predicted_calories = model.predict(user_data_scaled)[0]

    st.subheader("Predicted Calories Burned")
    st.write(f"{predicted_calories:.2f} kcal")

    # Save session history
    session_entry = {
        "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duration": duration,
        "Heart Rate": heart_rate,
        "Calories Burned": round(predicted_calories, 2)
    }
    st.session_state.session_history.append(session_entry)

    # Display session history
    st.subheader("Session History")
    history_df = pd.DataFrame(st.session_state.session_history)
    st.dataframe(history_df)

    # Progression Milestones
    st.subheader("Progression Milestones")
    if len(st.session_state.session_history) > 1:
        initial_weight = st.session_state.session_history[0]["Calories Burned"]
        latest_weight = st.session_state.session_history[-1]["Calories Burned"]
        progress = (latest_weight - initial_weight) / initial_weight * 100
        st.progress(min(100, max(0, int(progress + 50))))
    else:
        st.write("Complete more workouts to track progress!")

    # AI Chat Coach
    st.subheader("AI Chat Coach")
    user_question = st.text_input("Ask your fitness coach anything:")
    if user_question:
        response = "Stay consistent, eat healthy, and keep pushing!"  # Placeholder AI response
        st.write(f"Coach: {response}")

    # Personalized Recommendations
    st.subheader("Personalized Recommendations")
    if goal_type == "Weight Loss":
        st.write("Try incorporating more cardio exercises like running or cycling.")
    elif goal_type == "Muscle Gain":
        st.write("Increase protein intake and focus on strength training.")
    elif goal_type == "Endurance":
        st.write("Increase workout duration gradually and focus on stamina-building activities.")
    else:
        st.write("Maintain a balanced routine and stay active daily.")