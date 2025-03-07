import pandas as pd
from flask import Flask , request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
encoder_dict=pickle.load(open('encoders.pkl','rb'))


columns=["Car_Company","Model_Name","Location","Manufacturing_Year","Fuel_Type","Seats","Kms_Driven","Ownership","Transmission","Mileage (kmpl)","Engine (cc)","Max_Power (bhp)"]

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    # Get user input from form data
    data = request.form  # Assume input is sent as form data
    
    # Example input sent as form data:
    # feature1=A&feature2=5&feature3=B
    
    # Transform categorical features using encoders
    transformed_data = []
    for feature, value in data.items():
        value = value.title()  # Convert to title case
        if feature in encoder_dict:  # Check if the feature has an encoder
            encoder = encoder_dict[feature] 
            #transform returns a list and to extract first element [0] is used
            if value not in encoder.classes_:
                transformed_value = -1# Add unseen value
            else:
                transformed_value = encoder.transform([value])[0]
            transformed_data.append(transformed_value)
        else:
            # For numerical features or unencoded categorical features
            transformed_data.append(float(value))  # Ensure numerical values are converted properly
    # Convert to the format expected by the model (e.g., a NumPy array)
    transformed_df = pd.DataFrame([transformed_data], columns=columns)
    prediction = model.predict(transformed_df)
    prediction = f"{prediction[0]:.2f}"

    return render_template('index.html',prediction_text=(prediction))

if __name__ == '__main__':
    app.run(debug=True)
