# Here we will create our web application using f

from flask import Flask, request, render_template  # request for any get or post request from form
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler   # Srandard scaler for using pickle file from prediction pipeline

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)   # It is the entry point of excute the application

app=application  # Object of flask application

@app.route('/')   # It will indicate home page with '/'
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':                   # Get will give the form where we wil add our detail from which we want to predict the target value
        return render_template('home.html')
    else:
        data=CustomData(            # Here it is Post method where we have to do all the project pipeline work like EDA feature eagnineering, standard scaler etc. and then predict the value
            gender=request.form.get('gender'),   # Here request has all the information of input which we enter in the web apps so we are using request.form.set('gender') for givng gender data. Same way we can give for all trhe different column
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )   

        pred_df=data.get_data_as_data_frame()  # this will convert all the data to dataframe with this function
        print(pred_df)   

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)   # After creating objects of PredictPipeline it will call predict function of utils.py and give the results of prediction      
        return render_template('home.html',results=results[0])  # It will return the result on the home only. Taking results[0] as it will come is list form
    

if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)   # this host will mapp to local host 127.0.0.1

    

