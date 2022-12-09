
import pickle
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer


# path 
path = "./models"

# folder names where models live
folders = [
    "/call_to_action", 
    "/credibility_statement",
    "/greeting",
    "/intention_statement",
    "/intro",
    "/problem_statement",
    "/sign_off",
    "/value_prop",
    "/warm_up"

]

# model name
model_names = [
    "/linearSVC_clf.pkl",
    "/logReg_clf.pkl",
    "/multinomialNB_clf.pkl",
    "/randomForest_clf.pkl"
]


def predict_pipeline(pred_text):
    
    import statistics
    import operator

    path = "./models"
    
    folders = ["/call_to_action", "/credibility_statement","/greeting","/intention_statement",
                "/intro","/problem_statement","/sign_off","/value_prop","/warm_up"]
    
    label = [s.replace("/", "") for s in folders]

    model_names = ["/linearSVC_clf.pkl","/logReg_clf.pkl","/mutlinomialNB_clf.pkl","/randomForest_clf.pkl"]

    pred_confidence = []

    for folder in folders:
        # load tfidf
        tf1 = pickle.load(open(path+folder+ "/tfidf1.pkl", 'rb'))# loading dictionary
        # Create new tfidfVectorizer with old vocabulary
        tf1_new = TfidfVectorizer(sublinear_tf=True, norm='l2', 
                                encoding='latin-1', ngram_range=(1, 2), 
                                stop_words=None, vocabulary = tf1)# to use trained vectorizer, vocabulary= tf1 or loaded dict.
        # fit text you want to predict 
        X_tf1 = tf1_new.fit_transform([pred_text])

        # list of model results per category 
        res = []

        for model in model_names:
            clf = pickle.load(open(path + folder + model,'rb')) # loading model 
            res.append(clf.predict_proba(X_tf1)[0][1]) # predict

        pred = statistics.mean(res) # get average confindence from 4 models 

        pred_confidence.append(round(pred,4))
    

 
    # create dictionary of preds 
    preds = {label[i]: pred_confidence[i] for i in range(len(label))}
    # sort dictionary by values
    preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)}

    
    first = next(iter(preds.items()))
    prediction = first[0]
    confidence = (round((first[1] * 100), 2))

    

    # Make a list of text with sentiment.
    #data = []
    #data.append({'predictions': first, 'sentence': pred_text})

    data = []
    data.append({'prediction': prediction, 
                'confidence': confidence,
                 'sentence': pred_text,
                 'preds':preds})

    # return all instances of preds
    

    return data







