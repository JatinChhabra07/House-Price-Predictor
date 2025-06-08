import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv('data.csv')

# Prepare features and target
X = df[['Size', 'Bedrooms', 'Bathrooms', 'HouseAge', 'Location']]
y = df['Price']

# Preprocessing pipeline
categorical_features = ['Location']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Define prediction function
def predict_price():
    try:
        size = float(size_entry.get())
        bedrooms = int(bedroom_var.get())
        bathrooms = int(bathroom_var.get())
        house_age = int(age_var.get())
        location = location_var.get()

        input_data = pd.DataFrame([[size, bedrooms, bathrooms, house_age, location]],
                                  columns=['Size', 'Bedrooms', 'Bathrooms', 'HouseAge', 'Location'])

        predicted_price = pipeline.predict(input_data)[0]

        messagebox.showinfo("Predicted Price", f"Estimated House Price: ${predicted_price:,.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI Setup
root = tk.Tk()
root.title("House Price Predictor")

# Size Entry
tk.Label(root, text="House Size (sq ft):").grid(row=0, column=0, padx=10, pady=5)
size_entry = tk.Entry(root)
size_entry.grid(row=0, column=1, padx=10, pady=5)

# Bedrooms Dropdown
tk.Label(root, text="Bedrooms:").grid(row=1, column=0, padx=10, pady=5)
bedroom_var = tk.StringVar()
bedroom_combo = ttk.Combobox(root, textvariable=bedroom_var, values=[1, 2, 3, 4, 5], state='readonly')
bedroom_combo.grid(row=1, column=1, padx=10, pady=5)
bedroom_combo.current(0)

# Bathrooms Dropdown
tk.Label(root, text="Bathrooms:").grid(row=2, column=0, padx=10, pady=5)
bathroom_var = tk.StringVar()
bathroom_combo = ttk.Combobox(root, textvariable=bathroom_var, values=[1, 2, 3], state='readonly')
bathroom_combo.grid(row=2, column=1, padx=10, pady=5)
bathroom_combo.current(0)

# House Age Dropdown
tk.Label(root, text="House Age (years):").grid(row=3, column=0, padx=10, pady=5)
age_var = tk.StringVar()
age_combo = ttk.Combobox(root, textvariable=age_var, values=list(range(1, 11)), state='readonly')
age_combo.grid(row=3, column=1, padx=10, pady=5)
age_combo.current(0)

# Location Dropdown
tk.Label(root, text="Location:").grid(row=4, column=0, padx=10, pady=5)
location_var = tk.StringVar()
location_combo = ttk.Combobox(root, textvariable=location_var, values=['Suburb', 'City Center', 'Countryside'], state='readonly')
location_combo.grid(row=4, column=1, padx=10, pady=5)
location_combo.current(0)

# Predict Button
predict_btn = tk.Button(root, text="Predict Price", command=predict_price)
predict_btn.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

root.mainloop()
