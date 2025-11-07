# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')   # ✅ FIX: Disable GUI backend

import matplotlib.pyplot as plt
import os


app = Flask(__name__)

DATASET_PATH = "crop_data.csv"
MODEL_PATH = "model_pipeline.joblib"

# ✅ Load dataset
df_dataset = pd.read_csv(DATASET_PATH)

# ✅ Unique dropdown values (read directly from dataset)
unique_states = sorted(df_dataset["State"].unique())
unique_crops = sorted(df_dataset["Crop"].unique())
unique_seasons = sorted(df_dataset["Season"].unique())
unique_pesticides = sorted(df_dataset["Pesticide Name"].unique())
unique_soil_types = sorted(df_dataset["Soil Type"].unique())

# ✅ Build state → districts mapping
state_district_map = {}
for st in unique_states:
    state_district_map[st] = sorted(df_dataset[df_dataset["State"] == st]["District"].unique())


# ✅ Load ML model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except:
    model = None
    print("❌ MODEL NOT FOUND")


# ✅ Build Input
def build_input(data_dict):

    model_cols = [
        'State', 'District', 'Crop', 'Season', 'Year', 'Soil Type',
        'Area (acres)', 'Avg. Rainfall (mm)',
        'Pesticide Name', 'Avg. Temperature (°C)'
    ]

    mapping = {
        'state': 'State',
        'district': 'District',
        'crop': 'Crop',
        'season': 'Season',
        'crop_year': 'Year',
        'soil_type': 'Soil Type',
        'area': 'Area (acres)',
        'annual_rainfall': 'Avg. Rainfall (mm)',
        'pesticide_name': 'Pesticide Name',
        'avg_temperature': 'Avg. Temperature (°C)',
    }

    final_input = {}
    for form_key, col_name in mapping.items():
        val = data_dict.get(form_key)
        if col_name in ['State', 'District', 'Crop', 'Season', 'Pesticide Name', 'Soil Type']:
            final_input[col_name] = str(val)
        else:
            final_input[col_name] = float(val)

    df = pd.DataFrame([final_input], columns=model_cols)
    return df


# ✅ Recommend Best Fertilizer
def recommend_fertilizer(user_crop):
    crop_df = df_dataset[df_dataset["Crop"] == user_crop]

    if crop_df.empty:
        return None

    avg_yield = crop_df.groupby("Pesticide Name")["Yield (kg/acre)"].mean()
    best_pesticide = avg_yield.idxmax()

    return best_pesticide


# ✅ Home Page
@app.route('/')
def index():
    return render_template(
        'index.html',
        states=unique_states,
        crops=unique_crops,
        seasons=unique_seasons,
        pesticides=unique_pesticides,
        soil_types=unique_soil_types,
        state_district_map_json=state_district_map
    )


# ✅ Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: Model not loaded"

    form = request.form.to_dict()
    df_input = build_input(form)

    farmer_pred = model.predict(df_input)[0]
    farmer_pred = round(float(farmer_pred), 2)

    # ✅ Get recommended fertilizer
    best_fert = recommend_fertilizer(form["crop"])
    if best_fert is None:
        best_fert = form["pesticide_name"]

    # ✅ Predict with recommended fertilizer
    df_input_rec = df_input.copy()
    df_input_rec["Pesticide Name"] = best_fert
    rec_pred = model.predict(df_input_rec)[0]
    rec_pred = round(float(rec_pred), 2)

    # ✅ Graph – improvement bar chart
    labels = ["Farmer Fertilizer", "Recommended Fertilizer"]
    values = [farmer_pred, rec_pred]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, values)
    plt.ylabel("Yield (kg/acre)")
    plt.title("Yield Comparison")

    graph_path = "static/improvement.png"
    plt.savefig(graph_path)
    plt.close()

    total_farmer = round(farmer_pred * float(form["area"]), 2)
    total_rec = round(rec_pred * float(form["area"]), 2)

    return render_template(
        'result.html',
        farmer_fert=form["pesticide_name"],
        farmer_pred=farmer_pred,
        rec_fert=best_fert,
        rec_pred=rec_pred,
        total_farmer=total_farmer,
        total_rec=total_rec,
        graph_file=graph_path
    )


# ✅ Add Data Page
@app.route('/add-data')
def add_data():
    return render_template(
        "add_data.html",
        states=unique_states,
        crops=unique_crops,
        seasons=unique_seasons,
        pesticides=unique_pesticides,
        soil_types=unique_soil_types,
        state_district_map_json=state_district_map
    )


# ✅ Save New Data
@app.route("/save-data", methods=["POST"])
def save_data():

    try:
        df = pd.read_csv(DATASET_PATH)
    except:
        return "Dataset not found!", 404

    form = request.form
    new_row = {
        "State": form.get("state"),
        "District": form.get("district"),
        "Crop": form.get("crop"),
        "Season": form.get("season"),
        "Year": int(form.get("year")),
        "Soil Type": form.get("soil_type"),
        "Area (acres)": float(form.get("area")),
        "Avg. Rainfall (mm)": float(form.get("annual_rainfall")),
        "Pesticide Name": form.get("pesticide_name"),
        "Avg. Temperature (°C)": float(form.get("avg_temperature")),
        "Yield (kg/acre)": float(form.get("yield"))
    }

    # ✅ Replace deprecated append() with concat()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(DATASET_PATH, index=False)

    return render_template("success.html", message="✅ Data Added Successfully!")


# ✅ Run Server
if __name__ == "__main__":
    app.run(debug=True)
