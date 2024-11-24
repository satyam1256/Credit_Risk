from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('./Notebooks/bestmodel.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define feature fields
fields = [
    'enq_L3m', 'Age_Oldest_TL', 'num_std_12mts', 'pct_PL_enq_L6m_of_ever',
    'time_since_recent_enq', 'max_recent_level_of_deliq', 'recent_level_of_deliq',
    'PL_enq_L12m', 'Secured_TL', 'last_prod_enq2_ConsumerLoan', 'GL_Flag',
    'num_times_60p_dpd', 'num_deliq_6_12mts', 'Age_Newest_TL', 'PL_Flag'
]

@app.route('/')
def home():
    # Pass empty values initially
    return render_template('index.html', fields=fields, values={}, error=None, result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        values = {}
        for field in fields:
            if field == 'last_prod_enq2_ConsumerLoan':  # Handle boolean separately
                values[field] = request.form.get(field) == 'true'
            else:
                values[field] = float(request.form.get(field))  # Convert input to float for the model

        # Prepare input for the model
        input_features = [values[field] for field in fields]
        print(f"Input features: {input_features}")  # Debugging input values

        # Check if model is loaded
        if model is None:
            raise ValueError("Model not loaded.")

        # Make prediction
        prediction = model.predict([input_features])
        prediction_map = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
        result = prediction_map.get(prediction[0], "Unknown Prediction")

        # Render results
        return render_template('index.html', fields=fields, values=values, result=result, error=None)

    except Exception as e:
        # Render error
        return render_template('index.html', fields=fields, values=request.form, error=f"Error: {str(e)}", result=None)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")





# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the model
# model = pickle.load(open('bestmodel.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Collect all input fields from the form
#         enq_L3m = int(request.form['enq_L3m'])
#         Age_Oldest_TL = int(request.form['Age_Oldest_TL'])
#         num_std_12mts = int(request.form['num_std_12mts'])
#         pct_PL_enq_L6m_of_ever = float(request.form['pct_PL_enq_L6m_of_ever'])
#         time_since_recent_enq = int(request.form['time_since_recent_enq'])
#         max_recent_level_of_deliq = int(request.form['max_recent_level_of_deliq'])
#         recent_level_of_deliq = int(request.form['recent_level_of_deliq'])
#         PL_enq_L12m = int(request.form['PL_enq_L12m'])
#         Secured_TL = int(request.form['Secured_TL'])
#         last_prod_enq2_ConsumerLoan = bool(int(request.form['last_prod_enq2_ConsumerLoan']))
#         GL_Flag = int(request.form['GL_Flag'])
#         num_times_60p_dpd = int(request.form['num_times_60p_dpd'])
#         num_deliq_6_12mts = int(request.form['num_deliq_6_12mts'])
#         Age_Newest_TL = int(request.form['Age_Newest_TL'])
#         PL_Flag = int(request.form['PL_Flag'])

#         # Prepare input array
#         features = np.array([[
#             enq_L3m, Age_Oldest_TL, num_std_12mts, pct_PL_enq_L6m_of_ever,
#             time_since_recent_enq, max_recent_level_of_deliq, recent_level_of_deliq,
#             PL_enq_L12m, Secured_TL, last_prod_enq2_ConsumerLoan, GL_Flag,
#             num_times_60p_dpd, num_deliq_6_12mts, Age_Newest_TL, PL_Flag
#         ]])

#         # Make prediction
#         prediction = model.predict(features)[0]
#         result = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}.get(prediction, 'Unknown')

#         return render_template('index.html', result=result)

#     except Exception as e:
#         return render_template('index.html', error=f"An error occurred: {str(e)}")

# if __name__ == '__main__':
#     app.run(debug=True)
