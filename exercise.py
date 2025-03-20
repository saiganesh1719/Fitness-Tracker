import pandas as pd

# Create dummy data for calories.csv
calories_data = {
    "Age": [25, 30, 35, 40, 45],
    "Weight": [70, 80, 75, 85, 90],
    "Height": [175, 180, 170, 165, 160],
    "Duration": [30, 45, 60, 20, 50],
    "Heart_Rate": [120, 130, 140, 110, 125],
    "Calories": [250, 300, 350, 200, 275]
}

df_calories = pd.DataFrame(calories_data)
df_calories.to_csv("calories.csv", index=False)
print("✅ calories.csv saved successfully!")

# Create dummy data for exercise.csv
exercise_data = {
    "Exercise": ["Running", "Cycling", "Swimming", "Yoga", "Strength Training"],
    "Duration_Min": [30, 45, 60, 40, 50],
    "Calories_Burned": [300, 400, 500, 200, 350]
}

df_exercise = pd.DataFrame(exercise_data)
df_exercise.to_csv("exercise.csv", index=False)
print("✅ exercise.csv saved successfully!")
