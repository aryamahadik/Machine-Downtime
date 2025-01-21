# Clone the repository
git clone [https://github.com/aryamahadik/Machine-Downtime.git]
cd manufacturing-pred-api


# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py


make sure to create csv file here eg "sample_manufacturing_data.csv"

Go to http://localhost:5000/upload in your browser

Select your CSV file (must include Temperature, Run_Time, and Vibration columns) make changes in model if required

Click Upload

Step 2: Train Model
Train the model using your uploaded data:

use this for train in your terminal == curl -X POST -H "Content-Type: application/json" -d '{"filename":"sample_manufacturing_data.csv"}' http://localhost:5000/train

Step 3: Make Predictions
Make predictions with new machine data:

usethis for prediction in your terminal == curl -X POST -H "Content-Type: application/json" -d '{"Temperature":85.5,"Run_Time":145.0,"Vibration":0.68}' http://localhost:5000/predict


Dependencies

Flask
pandas
numpy
scikit-learn
joblib
werkzeug
