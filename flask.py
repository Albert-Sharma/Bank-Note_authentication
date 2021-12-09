from flask import Flask,request
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__) #the point we need to start the application

pickle_in = open('classifier.pkl','rb') #read-only in birany format
classifier = pickle.load(pickle_in)

@app.route('/') #route path
def welcome():
    return "Welcome All"

@app.route('/predict') #default get methods
def predit_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is"+ str(prediction)


@app.route('/predict_file',methods=["POST"])
def predit_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted values for the csv file is"+ str(list(prediction))


if __name__ == '__main__':
    app.run()