from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import logging

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Load the trained model
MODEL_PATH = r'C:\Users\SOUNDAR\Desktop\Miniproject\model\xgb_model3.pkl'

try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Ensure the file path is correct.")
    raise
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

# Mapping categorical inputs to numerical values
GENDER_MAP = {'Male': 1, 'Female': 0, 'Other': 2}
SMOKING_STATUS_MAP = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}
OBESITY_MAP = {'No': 0, 'Yes': 1}
DIABETES_MAP = {'No': 0, 'Yes': 1}
CHOLESTEROL_MAP = {'Normal': 0, 'High': 1}
STRESS_MAP = {'No': 0, 'Yes': 1}
COVID_MAP = {'No': 0, 'Yes': 1}
GENETIC_DISORDER_MAP = {'No': 0, 'Yes': 1}

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect form data
        name = request.form['name']
        prediction = "Stroke"  # Example of prediction
        return render_template('result.html', name=name, prediction=prediction)
    return render_template('index.html')

@app.route("/")
def home():
    """Render the HTML form."""
    return render_template("index.html")

@app.route('/aboutstroke')
def about_stroke():
    return render_template('aboutstroke.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and make predictions."""
    try:
        # Collect data from form
        data = request.form
        
        # Convert categorical features to numerical values using predefined mappings
        gender = GENDER_MAP.get(data["gender"], -1)  # Default to -1 if invalid
        smoking_status = SMOKING_STATUS_MAP.get(data["smoking_status"], -1)
        obesity = OBESITY_MAP.get(data["obesity"], -1)
        diabetes = DIABETES_MAP.get(data["diabetes"], -1)
        cholesterol = CHOLESTEROL_MAP.get(data["cholesterol"], -1)
        stress = STRESS_MAP.get(data["stress"], -1)
        covid_19 = COVID_MAP.get(data["covid_19"], -1)
        genetic_disorder = GENETIC_DISORDER_MAP.get(data["genetic_disorder"], -1)
        
        # Ensure all mappings are valid
        if -1 in [gender, smoking_status, obesity, diabetes, cholesterol, stress, covid_19, genetic_disorder]:
            return render_template("error.html", error="Invalid input provided.")
        
        input_data = {
            "gender": gender,
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": smoking_status,
            "obesity": obesity,
            "diabetes": diabetes,
            "cholesterol": cholesterol,
            "stress": stress,
            "covid_19": covid_19,
            "genetic_disorder": genetic_disorder,
            "weight": float(data["weight"]),
        }

        # Convert to DataFrame for model prediction
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
        result = "Stroke" if prediction == 1 else "No Stroke"

        return render_template("result.html", prediction=result)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
