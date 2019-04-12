# This file  predicts a paticipant show up probability .


import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import dataHelper

def loadModel():
    filename = 'randomForestTreeModel.dat'
    regressor = pickle.load(open(filename, 'rb'))
    return regressor

# Participant to be predicted. Define it yourself or 
# simply copy one from the dataset
# !!! Pay attention to the feature definition !!!
to_predict = np.array([[  9,10,20,200,9,3,1    ]]) # for this example, the regression should be 55

def processEvent(event):
    f = event['feature']
    
    toe = dataHelper.getToeVal( f.get("Type_of_event","") )
    d = f.get("Distance","")
    w = f.get("Warranty","")
    par = f.get("Participants","")
    pref = dataHelper.getEventPrefVal( f.get("Preference","") )
    prof = dataHelper.getProfessionVal( f.get("Profession","") )
    age = f.get("Age","")

    return np.array([[toe, d, w, par, pref, prof, age]])

def predictfrommodel(event, context):
    #print(event)
    f = event['feature']
    print(np.asarray(f))
    feat = processEvent(event)
    print(feat)

    RDTregressor = loadModel()

    number = float(RDTregressor.predict(feat)) 
    return { 'body': json.dumps(number)    }

# Uncomment this function for test purposes
def run():
  RDTregressor = loadModel()
  number = float(RDTregressor.predict(to_predict)) 
  print('new prediction:', number)
  event = { "feature": {
    "Type_of_event":"Workshop",
    "Distance" :10,
    "Warranty" :20,
    "Participants" :200,
    "Preference" :"Summit",
    "Profession" :"Student",
    "Age" :1}
   }

  res = predictfrommodel(event, "")
  print(res)

run()

#print('################################################################')
#print('########################   UBIETY    ###########################')
#print('################################################################')
#print('Predicted show up probability is:', float(RDTregressor.predict(to_predict)) )
#print('################################################################')
