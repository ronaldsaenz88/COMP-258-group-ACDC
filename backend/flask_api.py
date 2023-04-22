"""
# GROUP PROJECT

## Welcome to the COMP 258 Project - Neural Networks Project - Full Stack Application to predict first-year Persistence​

Relevant Information:

    College: Centennial College
    Program: Software Engineering Technology - Artificial Intelligence
    Term: Winter 2023
    Course: Neural Networks (SEC. 001) - COMP258001_2023W

Group AC/DC

Group Members

    Pereira Barbosa, Ana​
    Mina Trujillo, Joan​
    Radmy, Mahpara Rafia​
    Sidhu, Manipal​
    Saenz Huerta, Ronald​
    Massa Rebolledo, Stella

COMP 258 Project

Group Project

You are supposed to use Neural Networks and build a full-stack intelligent solution for any of the following problems:
-	student success in the program (program completion)
-	Persistence (1st year persistence)
-	Academic performance (intake GPA)
-	Other outcomes that you may discover in the dataset

Purpose:
The purpose of this project is to:
- Design and code full-stack intelligent apps using emerging frameworks
- Build a Rest or Graph QL API 
- Build a Front-End for the Rest/Graph QL API 
- Apply appropriate design patterns and principles
- Use Neural Networks to make intelligent use of data

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from os import path
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Your API definition
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AIModel():
    """ 
    AIModel Model: This class will open the model file, and the csv files.
    """

    def __init__(self):
        #TODO load the model
        self.model = tf.keras.models.load_model('my_last_model.h5', compile=False)

        # Load data
        self.deploy_folder = r'data'

        self.X_test_df = pd.read_csv(path.join(self.deploy_folder,"X_test_data_group_acdc.csv"))
        self.y_test_df = pd.read_csv(path.join(self.deploy_folder,"y_test_data_group_acdc.csv")).to_numpy()

        self.label_dict = {'FirstYearPersistence_no': 0, 'FirstYearPersistence_yes': 1}
        self.class_names = ['FirstTermGpa', 'SecondTermGpa', 'FirstLanguage', 'Funding', 'School', 'FastTrack', 'Coop', 'Residency', 'Gender', 'PreviousEducation', 'AgeGroup', 'HighSchoolAverageMark', 'MathScore', 'EnglishGrade']


@app.route("/api/summary", methods=['GET'])
#@cross_origin()
def summary():
    """ 
    Summary Method: This method returns the summary of the model created before using Tensorflow and Keras.
    """

    # AI Model object
    ai_model = AIModel()

    # Empty list
    string_list = []

    # Get Summary of the AI model (Tensorflow, Keras) and fill the list string_list LIST
    ai_model.model.summary(line_length=80, print_fn=lambda x: string_list.append(x))

    # Transform the list into string variable
    summary_json = "\n".join(string_list)

    return jsonify({'output':summary_json})


@app.route("/api/scores", methods=['GET','POST']) #use decorator pattern for the route
#@cross_origin()
def scores():
    """ 
    Scores Method: This method returns the stats or scores of the model using the Test Data splitted before in the AI developed.
    """

    # AI Model object
    ai_model = AIModel()

    # Predict the Test data
    y_pred = ai_model.model.predict(ai_model.X_test_df)

    # Label predict values
    predict_labeled = np.argmax(y_pred, axis=1)

    # Scores / Stats of the model compared the Test Data with the predict values
    accuracy = accuracy_score(ai_model.y_test_df.argmax(axis=1), predict_labeled)
    precision = precision_score(ai_model.y_test_df.argmax(axis=1), predict_labeled)
    recall = recall_score(ai_model.y_test_df.argmax(axis=1), predict_labeled)
    f1 = f1_score(ai_model.y_test_df.argmax(axis=1), predict_labeled)
    roc_auc = roc_auc_score(ai_model.y_test_df.argmax(axis=1), predict_labeled)
    conf_matrix = confusion_matrix(ai_model.y_test_df.argmax(axis=1), predict_labeled)         
    classification_report_var = classification_report(ai_model.y_test_df.argmax(axis=1), predict_labeled, target_names=ai_model.label_dict)

    # Json Response with all stats/scores
    resJson = {
        "accuracy": accuracy,
        "precision": precision,
        "recall":recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confussion_matrix": str(conf_matrix),
        "classification_report_var": classification_report_var
    }

    return jsonify({'output':resJson})


@app.route("/api/predict", methods=['POST'])
#@cross_origin()
def predict():
    """ 
    Predict Method: This method returns the prediction of the model or scores of the model using the Test Data splitted before in the AI developed.
    """

    # AI Model object
    ai_model = AIModel()

    # Get values of the POST DATA (Json)
    json_dict = request.json

    # Convert the json dict into Pandas Dataframe
    df = pd.DataFrame(json_dict, columns=ai_model.class_names)

    # Change data type
    df["FirstTermGpa"] = df["FirstTermGpa"].astype(np.float64)
    df["SecondTermGpa"] = df["SecondTermGpa"].astype(np.float64)
    df["Funding"] = df["Funding"].astype(np.float64)
    df["School"] = df["School"].astype(np.float64)
    df["FastTrack"] = df["FastTrack"].astype(np.float64)
    df["Coop"] = df["Coop"].astype(np.float64)
    df["Residency"] = df["Residency"].astype(np.float64)
    df["Gender"] = df["Gender"].astype(np.float64)
    df["PreviousEducation"] = df["PreviousEducation"].astype(np.float64)
    df["AgeGroup"] = df["AgeGroup"].astype(np.float64)
    df["HighSchoolAverageMark"] = df["HighSchoolAverageMark"].astype(np.float64)
    df["MathScore"] = df["MathScore"].astype(np.float64)
    df["EnglishGrade"] = df["EnglishGrade"].astype(np.float64)

    # Put into list
    row = [df]

    # Get Prediction of this post data
    y_predict = ai_model.predict(df)

    return json.dumps({'output':(y_predict.argmax(axis=1))}, cls=NumpyEncoder)


#Main method, runs at the very beginning...
if __name__ == '__main__':
    port = 5000 # If you don't provide any port the port will be set to 12345

    app.run(host="127.0.0.1", port=port, debug=True) #Run the app
