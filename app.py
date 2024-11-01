from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Collecting all the input values from the form
        v1 = request.form.get('pct_tl_open_L6M', type=float)
        v2 = request.form.get('pct_tl_closed_L6M', type=float)
        v3 = request.form.get('Tot_TL_closed_L12M', type=float)
        v4 = request.form.get('pct_tl_closed_L12M', type=float)
        v5 = request.form.get('Tot_Missed_Pmnt', type=float)
        v6 = request.form.get('CC_TL', type=float)
        v7 = request.form.get('Home_TL', type=float)
        v8 = request.form.get('PL_TL', type=float)
        v9 = request.form.get('Secured_TL', type=float)
        v10 = request.form.get('Unsecured_TL', type=float)
        v11 = request.form.get('Other_TL', type=float)
        v12 = request.form.get('Age_Oldest_TL', type=float)
        v13 = request.form.get('Age_Newest_TL', type=float)
        v14 = request.form.get('time_since_recent_payment', type=float)
        v15 = request.form.get('max_recent_level_of_deliq', type=float)
        v16 = request.form.get('num_deliq_6_12mts', type=float)
        v17 = request.form.get('num_times_60p_dpd', type=float)
        v18 = request.form.get('num_std_12mts', type=float)
        v19 = request.form.get('num_sub', type=float)
        v20 = request.form.get('num_sub_6mts', type=float)
        v21 = request.form.get('num_sub_12mts', type=float)
        v22 = request.form.get('num_dbt', type=float)
        v23 = request.form.get('num_dbt_12mts', type=float)
        v24 = request.form.get('num_lss', type=float)
        v25 = request.form.get('recent_level_of_deliq', type=float)
        v26 = request.form.get('CC_enq_L12m', type=float)
        v27 = request.form.get('PL_enq_L12m', type=float)
        v28 = request.form.get('time_since_recent_enq', type=float)
        v29 = request.form.get('enq_L3m', type=float)
        v30 = request.form.get('NETMONTHLYINCOME', type=float)
        v31 = request.form.get('Time_With_Curr_Empr', type=float)
        v32 = request.form.get('CC_Flag', type=int)
        v33 = request.form.get('PL_Flag', type=int)
        v34 = request.form.get('pct_PL_enq_L6m_of_ever', type=float)
        v35 = request.form.get('pct_CC_enq_L6m_of_ever', type=float)
        v36 = request.form.get('HL_Flag', type=int)
        v37 = request.form.get('GL_Flag', type=int)
        v38 = request.form.get('EDUCATION', type=int)
        v39 = request.form.get('MARITALSTATUS_Married', type=int)
        v40 = request.form.get('MARITALSTATUS_Single', type=int)
        v41 = request.form.get('GENDER_F', type=int)
        v42 = request.form.get('GENDER_M', type=int)
        v43 = request.form.get('last_prod_enq2_AL', type=int)
        v44 = request.form.get('last_prod_enq2_CC', type=int)
        v45 = request.form.get('last_prod_enq2_ConsumerLoan', type=int)
        v46 = request.form.get('last_prod_enq2_HL', type=int)
        v47 = request.form.get('last_prod_enq2_PL', type=int)
        v48 = request.form.get('last_prod_enq2_others', type=int)
        v49 = request.form.get('first_prod_enq2_AL', type=int)
        v50 = request.form.get('first_prod_enq2_CC', type=int)
        v51 = request.form.get('first_prod_enq2_ConsumerLoan', type=int)
        v52 = request.form.get('first_prod_enq2_HL', type=int)
        v53 = request.form.get('first_prod_enq2_PL', type=int)
        v54 = request.form.get('first_prod_enq2_others', type=int)

        # Make prediction
        result = model.predict([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54]])[0]

        # Pass input values and result back to the template
        return render_template('index.html', result=result, **locals())
    else:
        # If GET request, redirect to home page
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
