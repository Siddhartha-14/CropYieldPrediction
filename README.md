# 🌾 Crop Yield Prediction System — Version 1.0.0

A Machine Learning–powered web application built with **Flask**, designed to predict crop yield based on environmental and agricultural features.
This project helps farmers, researchers, and agricultural departments make data-driven decisions for improved productivity.

---

## ✅ **Features (v1.0.0)**

### 🔮 **1. Machine Learning Prediction Engine**

* Built using **XGBoost + Scikit-learn Pipelines**
* Predicts yield (kg/acre) using:

  * State & District
  * Crop
  * Season
  * Soil Type
  * Area
  * Rainfall (mm)
  * Temperature (°C)
  * Pesticide
* Fast inference and optimized preprocessing

### 🎨 **2. Modern UI with TailwindCSS**

* Responsive design
* Smooth animations
* Form validation
* Clear mandatory labels

### 🏙️ **3. Dynamic State → District Auto-Fill**

* Automatically updates district dropdown based on state
* Prevents invalid location inputs

### ➕ **4. Add Dataset Record Page**

* Add new crop records manually
* Same modern UI as index page
* Clean input layout with icons
* Supports all fields needed by ML model

### ✅ **5. Prediction Analysis**

* Shows predicted yield
* Generates comparison graph
* Clean result page with visualization

### ✅ **6. Success Page**

* Displays confirmation message after adding new records
* Buttons: Go Home / Add More Data

---

# 📁 **Project Structure**

```
CropYieldPrediction/
│── app.py
│── train_model.py
│── crop_data.csv
│── model_pipeline.joblib
│── /static
│     ├── improvement.png
│── /templates
│     ├── index.html
│     ├── result.html
│     ├── add_data.html
│     ├── success.html
│── README.md
```

---

# 🛠️ **Tech Stack**

### **Backend**

* Python
* Flask
* Scikit-learn
* XGBoost
* Pandas
* NumPy

### **Frontend**

* HTML
* TailwindCSS
* JavaScript
* Jinja2 Templates

### **Storage**

* CSV (default)
* (DB integration planned for v2.0)

---

# 🚀 **How to Run Locally**

### ✅ **1. Clone Repository**

```
git clone https://github.com/Siddhartha-14/CropYieldPrediction.git
cd CropYieldPrediction
```

### ✅ **2. Create Virtual Environment**

```
python -m venv venv
venv\Scripts\activate     # Windows
```

### ✅ **3. Install Dependencies**

```
pip install -r requirements.txt
```

### ✅ **4. Run Flask App**

```
python app.py
```

App will start at:

```
http://127.0.0.1:5000
```

---

# 📊 **Model Training**

To retrain the model:

```
python train_model.py
```

This will update:

✅ `model_pipeline.joblib`
✅ `crop_data.csv` (if appended)

---

# 🖼️ **Screenshots**

✅ *Add your screenshots here (UI, prediction result, dataset form)*

---

# 🏷️ **Version 1.0.0 — Highlights**

* ✅ First stable release
* ✅ Fully working ML predictions
* ✅ Improved UI with Tailwind
* ✅ Add data + success workflow
* ✅ Dynamic dropdown logic
* ✅ Graph visualization


