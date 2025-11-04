# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from xgboost import XGBRegressor   # ✅ XGBoost added

# Use the correct CSV name
CSV_PATH = "crop_data.csv"
MODEL_PATH = "model_pipeline.joblib"

def load_and_prepare(df):
    df = df.rename(columns={'Yield (kg/acre)': 'target_yield'})
    df = df.dropna(subset=['target_yield'])

    feature_cols = [
        'State', 'District', 'Crop', 'Season', 'Year', 'Soil Type',
        'Area (acres)', 'Avg. Rainfall (mm)',
        'Pesticide Name', 'Avg. Temperature (°C)'
    ]

    X = df[feature_cols]
    y = df['target_yield'].astype(float)
    return X, y

def build_pipeline(categorical_cols, numeric_cols):
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat, categorical_cols),
            ('num', num, numeric_cols)
        ],
        remainder='drop'
    )

    # ✅ XGBoost Regressor
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )

    return Pipeline([
        ('pre', preprocessor),
        ('model', model)
    ])

if __name__ == "__main__":
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: The file {CSV_PATH} was not found.")
        exit()

    X, y = load_and_prepare(df)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(categorical_cols, numeric_cols)

    print("🚀 Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    print("✅ Training completed.")

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Evaluation:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved as {MODEL_PATH}")
