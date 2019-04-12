# This the source code for the random forest algorithm developed for the EXIST MVP

## Depenencies 
  - numpy
  - sklearn
  - pickle
  - random
  

## Data set generation 
The actual provided dataset has been artificially generated using different probabilities distributions in order to approximate real life expectations for every single feature.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies.

Run 
```bash
python dataGen.py -s 
```
or 
```bash
python dataGen.py -a
```
for generating a new dataset with a single event type or multiple event types respectively.
Consider also changing the number of sample to be generated. Default is 5000

## Training 

Run 
```bash
python Exist_RF_Algorithm.py -n 20 -s
```
for training the random forest algorithm with cross validation. -n specifies the number of folds (n=20) on the dataset and its default value is 10. Set the flag -s to save the trained tree as a file for later predictions

Note: The best tree when cross validating is selected and returned or saved by the algorithm

## Prediction 

Run 
```bash
python predict.py 
```
in order to predict the show up probability of a user. A default user is set by default. Consider changing the features for further prediction. A batch of users can also be predicted.

## Licenese 
[MIT](https://choosealicense.com/licenses/mit/)
