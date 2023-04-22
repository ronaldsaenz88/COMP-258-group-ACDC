from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import keras
import sys
import numpy as np
import pandas as pd
import json
from os import path
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# Your API definition
#app = Flask(__name__)
#CORS(app)

# Your API definition
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#TODO load the model
model = tf.keras.models.load_model('my_last_model.h5', compile=False)

# Load data
deploy_folder = r'data'

X_test_df = pd.read_csv(path.join(deploy_folder,"X_test_data_group_acdc.csv"))
y_test_df = pd.read_csv(path.join(deploy_folder,"y_test_data_group_acdc.csv")).to_numpy()


@app.route("/api/summary", methods=['GET'])
#@cross_origin()
def summary():
    string_list = []
    model.summary(line_length=80, print_fn=lambda x: string_list.append(x))
    summary_json = "\n".join(string_list)
    
    return jsonify({'output':summary_json})



@app.route("/api/scores", methods=['GET','POST']) #use decorator pattern for the route
#@cross_origin()
def scores():              
    y_pred = model.predict(X_test_df)
    
    predict_labeled = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test_df.argmax(axis=1), predict_labeled)
    precision = precision_score(y_test_df.argmax(axis=1), predict_labeled)
    recall = recall_score(y_test_df.argmax(axis=1), predict_labeled)
    f1 = f1_score(y_test_df.argmax(axis=1), predict_labeled)
    roc_auc = roc_auc_score(y_test_df.argmax(axis=1), predict_labeled)
    conf_matrix = confusion_matrix(y_test_df.argmax(axis=1), predict_labeled)         

    label_dict = {'FirstYearPersistence_no': 0, 'FirstYearPersistence_yes': 1}

    classification_report_var = classification_report(y_test_df.argmax(axis=1), predict_labeled, target_names=label_dict)

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
    json_dict = request.json
    class_names = ['FirstTermGpa', 'SecondTermGpa', 'FirstLanguage', 'Funding', 'School', 'FastTrack', 'Coop', 'Residency', 'Gender', 'PreviousEducation', 'AgeGroup', 'HighSchoolAverageMark', 'MathScore', 'EnglishGrade']
 
    df = pd.DataFrame(json_dict, columns=class_names)

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

    #row = [[4, 3, 1, 1, 6, 2, 1, 1, 1, 2, 3, 92, 41, 9]]
    row = [df]
    y_hat = model.predict(df)

    return json.dumps({'output':(y_hat.argmax(axis=1))}, cls=NumpyEncoder)


#Main method, runs at the very beginning...
if __name__ == '__main__':
    port = 12345 # If you don't provide any port the port will be set to 12345
        
    app.run(host="0.0.0.0", port=port, debug=True) #Run the app
    

