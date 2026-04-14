from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import shap
import base64
import io
import os
import traceback
import matplotlib

# Use a non-GUI backend to avoid tkinter/main-loop errors when running under Flask
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Get the directory of the current file (app.py)
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Build model paths
MODEL_PATH = os.path.join(APP_DIR, "lightgbm_model.pkl")
SCALER_PATH = os.path.join(APP_DIR, "scaler.pkl")
IMPUTER_PATH = os.path.join(APP_DIR, "imputer.pkl")

print(f"Looking for models in: {APP_DIR}")
print(f"Model path: {MODEL_PATH}")

# Load models and preprocessors
try:
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model file not found: {MODEL_PATH}")
        model = None
    else:
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    if not os.path.exists(SCALER_PATH):
        print(f"✗ Scaler file not found: {SCALER_PATH}")
        scaler = None
    else:
        scaler = joblib.load(SCALER_PATH)
        print(f"✓ Scaler loaded from {SCALER_PATH}")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")
    scaler = None

try:
    if not os.path.exists(IMPUTER_PATH):
        print(f"✗ Imputer file not found: {IMPUTER_PATH}")
        imputer = None
    else:
        imputer = joblib.load(IMPUTER_PATH)
        print(f"✓ Imputer loaded from {IMPUTER_PATH}")
except Exception as e:
    print(f"✗ Error loading imputer: {e}")
    imputer = None

if model and scaler and imputer:
    print("✓ All models loaded successfully")
else:
    print("✗ Failed to load all models")


def get_feature_names():
    """Return feature names in correct order"""
    return [
        "Age",
        "Gender",
        "Total_Bilirubin",
        "Direct_Bilirubin",
        "Alkaline_Phosphotase",
        "Alamine_Aminotransferase",
        "Aspartate_Aminotransferase",
        "Total_Protiens",
        "Albumin",
        "Albumin_and_Globulin_Ratio",
    ]


def generate_shap_plot(shap_values, X_sample):
    """Generate SHAP summary plot and return as base64"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create simple SHAP value bar plot
        if hasattr(shap_values, "values"):
            sv = shap_values.values[0]
        elif isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
        else:
            sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        feature_names = get_feature_names()
        indices = np.argsort(np.abs(sv))[-5:]  # Top 5

        ax.barh(range(len(indices)), sv[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("SHAP Value")
        ax.set_title("Top 5 Feature Contributions")
        plt.tight_layout()

        # Convert to base64
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return plot_url
    except Exception as e:
        print(f"Error generating SHAP plot: {e}")
        return None


def generate_shap_force_plot(shap_values, X_sample):
    """Generate SHAP feature importance plot"""
    try:
        # Extract SHAP values
        if hasattr(shap_values, "values"):
            sv = shap_values.values[0]
        elif isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values[1].shape) > 1 else shap_values[1]
        else:
            sv = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        feature_names = get_feature_names()

        # Create matplotlib plot
        fig = plt.figure(figsize=(10, 6))
        indices = np.argsort(np.abs(sv))[-10:]  # Top 10

        colors = ["red" if x > 0 else "blue" for x in sv[indices]]
        plt.barh(range(len(indices)), sv[indices], color=colors, alpha=0.6)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.title("Feature Importance - SHAP Values")
        plt.tight_layout()

        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight", dpi=100)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return plot_url
    except Exception as e:
        print(f"Error generating force plot: {e}")
        return None


def get_shap_feature_importance(shap_values):
    """Get feature importance from SHAP values"""
    try:
        feature_names = get_feature_names()
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": mean_abs_shap}
        ).sort_values("Importance", ascending=False)

        return importance_df.to_dict("records")
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        return []


@app.route("/")
def home():
    """Render home page"""
    # Model name and metrics
    model_info = {
        "name": "LightGBM Classifier",
        "accuracy": 74.00,
        "precision": 76.53,
        "recall": 90.36,
        "f1_score": 82.87,
    }
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["POST"])
def predict():
    """Make prediction with SHAP explanation"""
    try:
        # Check if models are loaded
        if model is None or scaler is None or imputer is None:
            return jsonify({"error": "Models not loaded", "success": False}), 500

        # Get input data
        data = request.json
        if data is None:
            return jsonify({"error": "No JSON data provided", "success": False}), 400

        feature_names = get_feature_names()

        # Create input array
        try:
            input_data = np.array(
                [
                    float(data.get("age", 0)),
                    float(data.get("gender", 0)),
                    float(data.get("total_bilirubin", 0)),
                    float(data.get("direct_bilirubin", 0)),
                    float(data.get("alkaline_phosphotase", 0)),
                    float(data.get("alamine_aminotransferase", 0)),
                    float(data.get("aspartate_aminotransferase", 0)),
                    float(data.get("total_protiens", 0)),
                    float(data.get("albumin", 0)),
                    float(data.get("albumin_and_globulin_ratio", 0)),
                ]
            ).reshape(1, -1)
        except (ValueError, TypeError) as e:
            return (
                jsonify({"error": f"Invalid data types: {str(e)}", "success": False}),
                400,
            )

        # Validate input
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            return (
                jsonify(
                    {"error": "Invalid input values (NaN or Inf)", "success": False}
                ),
                400,
            )

        # Preprocess
        try:
            # The imputer was trained with 11 features (10 features + Dataset column)
            # So we need to add a dummy Dataset column before imputing
            # Create a copy with the dummy column
            input_with_dummy = np.hstack(
                [
                    input_data,  # 10 features
                    np.zeros((input_data.shape[0], 1)),  # Add dummy Dataset column
                ]
            )

            # Now impute (will work with 11 features)
            input_imputed = imputer.transform(input_with_dummy)

            # Remove the dummy column after imputing (only use first 10 features)
            input_imputed = input_imputed[:, :-1]

            # Now scale (scaler expects 10 features)
            input_scaled = scaler.transform(input_imputed)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            import traceback

            traceback.print_exc()
            return (
                jsonify(
                    {"error": f"Data preprocessing failed: {str(e)}", "success": False}
                ),
                400,
            )

        # Make prediction
        try:
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        except Exception as e:
            print(f"Model prediction error: {e}")
            return (
                jsonify(
                    {"error": f"Model prediction failed: {str(e)}", "success": False}
                ),
                400,
            )

        disease_probability = float(probability[1]) * 100
        healthy_probability = float(probability[0]) * 100

        # Get SHAP explanation using Explainer instead of TreeExplainer
        shap_plot = None
        feature_importance = []

        try:
            # Use shap.Explainer which works better with LGBMClassifier
            # Use the preprocessed input as background data (limited but better than nothing)
            explainer = shap.Explainer(model)
            shap_values = explainer(input_scaled)

            # Get feature contributions
            for i, feature in enumerate(feature_names):
                if hasattr(shap_values, "values"):
                    shap_value = shap_values.values[0, i]
                elif isinstance(shap_values, list):
                    shap_value = (
                        shap_values[1][0, i]
                        if prediction == 1
                        else shap_values[0][0, i]
                    )
                else:
                    shap_value = (
                        shap_values[0, i]
                        if len(shap_values.shape) > 1
                        else shap_values[i]
                    )

                feature_importance.append(
                    {
                        "feature": feature,
                        "value": float(input_scaled[0, i]),
                        "shap_value": float(shap_value),
                        "contribution": (
                            "Positive" if float(shap_value) > 0 else "Negative"
                        ),
                    }
                )

            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            # Generate SHAP visualization (optional, don't fail if it errors)
            try:
                shap_plot = generate_shap_force_plot(shap_values, input_scaled)
            except Exception as plot_error:
                print(f"Warning: SHAP plot generation failed: {plot_error}")
                shap_plot = None

        except Exception as shap_error:
            print(f"Warning: SHAP explanation failed: {shap_error}")
            # Continue without SHAP if it fails - still provide prediction
            for i, feature in enumerate(feature_names):
                feature_importance.append(
                    {
                        "feature": feature,
                        "value": float(input_scaled[0, i]),
                        "shap_value": 0.0,
                        "contribution": "Unknown",
                    }
                )

        # Prepare response
        response = {
            "success": True,
            "prediction": "Liver Disease" if prediction == 1 else "No Liver Disease",
            "prediction_class": int(prediction),
            "disease_probability": round(disease_probability, 2),
            "healthy_probability": round(healthy_probability, 2),
            "confidence": round(max(disease_probability, healthy_probability), 2),
            "feature_importance": feature_importance[:5],  # Top 5
            "all_features": feature_importance,
            "shap_plot": shap_plot,
        }

        return jsonify(response)

    except Exception as e:
        import traceback

        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {str(e)}", "success": False}), 400


@app.route("/api/shap-summary", methods=["POST"])
def shap_summary():
    """Get SHAP feature importance summary"""
    try:
        if model is None or scaler is None or imputer is None:
            return jsonify({"error": "Models not loaded", "success": False}), 500

        data = request.json
        if data is None:
            return jsonify({"error": "No JSON data provided", "success": False}), 400

        feature_names = get_feature_names()

        # Create input array
        try:
            input_data = np.array(
                [
                    float(data.get("age", 0)),
                    float(data.get("gender", 0)),
                    float(data.get("total_bilirubin", 0)),
                    float(data.get("direct_bilirubin", 0)),
                    float(data.get("alkaline_phosphotase", 0)),
                    float(data.get("alamine_aminotransferase", 0)),
                    float(data.get("aspartate_aminotransferase", 0)),
                    float(data.get("total_protiens", 0)),
                    float(data.get("albumin", 0)),
                    float(data.get("albumin_and_globulin_ratio", 0)),
                ]
            ).reshape(1, -1)
        except (ValueError, TypeError) as e:
            return (
                jsonify({"error": f"Invalid data types: {str(e)}", "success": False}),
                400,
            )

        # Preprocess
        try:
            # The imputer was trained with 11 features (10 features + Dataset column)
            # Add dummy Dataset column before imputing
            input_with_dummy = np.hstack(
                [
                    input_data,  # 10 features
                    np.zeros((input_data.shape[0], 1)),  # Add dummy Dataset column
                ]
            )

            # Impute with 11 features
            input_imputed = imputer.transform(input_with_dummy)

            # Remove dummy column (only use first 10 features)
            input_imputed = input_imputed[:, :-1]

            # Scale (expects 10 features)
            input_scaled = scaler.transform(input_imputed)
        except Exception as e:
            print(f"Preprocessing error: {e}")
            import traceback

            traceback.print_exc()
            return (
                jsonify(
                    {"error": f"Data preprocessing failed: {str(e)}", "success": False}
                ),
                400,
            )

        # Get SHAP values using Explainer
        try:
            explainer = shap.Explainer(model, input_scaled[:1])
            shap_values = explainer(input_scaled)

            return jsonify(
                {
                    "success": True,
                    "shap_values": (
                        shap_values.values.tolist()
                        if hasattr(shap_values, "values")
                        else (
                            shap_values.tolist()
                            if isinstance(shap_values, np.ndarray)
                            else [sv.tolist() for sv in shap_values]
                        )
                    ),
                }
            )
        except Exception as e:
            print(f"SHAP calculation error: {e}")
            return (
                jsonify(
                    {"error": f"SHAP calculation failed: {str(e)}", "success": False}
                ),
                400,
            )

    except Exception as e:
        import traceback

        print(f"Error in shap_summary: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Request failed: {str(e)}", "success": False}), 400


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
