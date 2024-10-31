import pickle
from flask import Flask, render_template, request, jsonify

# Load your machine learning model
with open('grid_search.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the POST request
    data = request.get_json()

    # Extract all the necessary columns from the data
    input_data = [
        data['pct_tl_open_L6M'], data['pct_tl_closed_L6M'], data['Tot_TL_closed_L12M'],
        data['pct_tl_closed_L12M'], data['Tot_Missed_Pmnt'], data['CC_TL'], data['Home_TL'],
        data['PL_TL'], data['Secured_TL'], data['Unsecured_TL'], data['Other_TL'],
        data['Age_Oldest_TL'], data['Age_Newest_TL'], data['time_since_recent_payment'],
        data['max_recent_level_of_deliq'], data['num_deliq_6_12mts'], data['num_times_60p_dpd'],
        data['num_std_12mts'], data['num_sub'], data['num_sub_6mts'], data['num_sub_12mts'],
        data['num_dbt'], data['num_dbt_12mts'], data['num_lss'], data['recent_level_of_deliq'],
        data['CC_enq_L12m'], data['PL_enq_L12m'], data['time_since_recent_enq'],
        data['enq_L3m'], data['NETMONTHLYINCOME'], data['Time_With_Curr_Empr'], data['CC_Flag'],
        data['PL_Flag'], data['pct_PL_enq_L6m_of_ever'], data['pct_CC_enq_L6m_of_ever'],
        data['HL_Flag'], data['GL_Flag'], data['EDUCATION'], data['MARITALSTATUS_Married'],
        data['MARITALSTATUS_Single'], data['GENDER_F'], data['GENDER_M'],
        data['last_prod_enq2_AL'], data['last_prod_enq2_CC'], data['last_prod_enq2_ConsumerLoan'],
        data['last_prod_enq2_HL'], data['last_prod_enq2_PL'], data['last_prod_enq2_others'],
        data['first_prod_enq2_AL'], data['first_prod_enq2_CC'], data['first_prod_enq2_ConsumerLoan'],
        data['first_prod_enq2_HL'], data['first_prod_enq2_PL'], data['first_prod_enq2_others']
    ]

    # Make prediction using the model
    prediction = model.predict([input_data])

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)



from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF

app = Flask(_name_)

# Load pre-trained models and scaler
scaler = joblib.load('models/scaler.pkl')
models = {
    'SVM': joblib.load('models/model_svm.pkl'),
    'Decision Tree': joblib.load('models/model_decision_tree.pkl'),
    'Random Forest': joblib.load('models/model_random_forest.pkl')
}

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting form data
        rock_strength = float(request.form['rock_strength'])
        pillar_width = float(request.form['pillar_width'])
        pillar_height = float(request.form['pillar_height'])
        mining_depth = float(request.form['mining_depth'])
        gallery_width = float(request.form['gallery_width'])

        # Creating a DataFrame from the input
        input_data = pd.DataFrame({
            'Rock_Strength': [rock_strength],
            'Pillar_Width': [pillar_width],
            'Pillar_Height': [pillar_height],
            'Mining_Depth': [mining_depth],
            'Gallery_Width': [gallery_width]
        })

        # Preprocess the data (scale)
        new_data_scaled = scaler.transform(input_data)

        # Predict stability using all models
        predictions = {}
        for model_name, model in models.items():
            prediction = model.predict(new_data_scaled)[0]
            stability_status = "Stable" if prediction == 1 else "Not Stable"
            predictions[model_name] = stability_status

        # Check if the PDF generation button was pressed
        if 'generate_pdf' in request.form:
            generate_pdf(input_data, predictions)

        return render_template('index.html', 
                               predictions=predictions,
                               rock_strength=rock_strength,
                               pillar_width=pillar_width,
                               pillar_height=pillar_height,
                               mining_depth=mining_depth,
                               gallery_width=gallery_width)
    
    except Exception as e:
        return str(e)

# Function to generate PDF report
# def generate_pdf(input_data, predictions):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", size=12)

#     pdf.cell(200, 10, txt="Pillar Stability Prediction Report", ln=True, align='C')
#     pdf.cell(200, 10, txt="", ln=True)  # Blank line

#     # Add input parameters
#     pdf.cell(200, 10, txt="Input Parameters:", ln=True)
#     for key, value in input_data.iloc[0].items():
#         pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

#     pdf.cell(200, 10, txt="", ln=True)  # Blank line

#     # Add predictions
#     pdf.cell(200, 10, txt="Prediction Results:", ln=True)
#     for model, prediction in predictions.items():
#         pdf.cell(200, 10, txt=f"{model}: {prediction}", ln=True)

#     pdf.output("report.pdf")

# if _name_ == '_main_':
#     app.run(debug=True)