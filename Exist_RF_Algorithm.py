# This file trains the random forest trees for predicting 
# the show up probability of an event participant.
# N-fold cross validation is used for choosing the tree having the best performance. 
# The number of folds must be specified at training time. This is set to default value 10
# 

# Training dependencies
import numpy as np 
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pickle 
from random import randrange
import random as rd
import argparse
import sys

#import sklearn 
#print('The scikit-learn version is {}.'.format(sklearn.__version__))

rd.seed(10)

data = np.genfromtxt('simple_data.csv', delimiter=',')
# remove the header
data = np.delete(data, 0,0) 

# split data into train and test samples 80% - 20% 
Xtrain= np.asarray(data[:int(len(data)*0.8)])
#Ytrain = np.asarray(labels[:int(len(data)*0.8)])

# test data
Xtest =  np.asarray(data[int(len(data)*0.8): len(data)])[:,[0,1,2,3,4,5,6]]
# test label 
Ytest =  np.asarray(data[int(len(data)*0.8): len(data)])[:,7]   

#print(list(Xtrain.dtype.names))
#print(labels.shape)
#print(training_data.shape)

def cross_validation_split(dataset, n_folds):
    '''Splits the data for cross validation'''
    dataset_split = list()

    # make a copy of the dataset as we iteratively remove object from it 
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()

        # randomly pick a particpant from the dataset
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        # append new generated fold
        dataset_split.append(fold)
    return dataset_split




def accuracy_metric(actual, predicted):
    '''Calculate accuracy percentage with 5% confidence '''
    correct = 0
    #for each actual label
    for i in range(len(actual)):
        #if actual matches predicted label by 5%
        if actual[i] >= predicted[i] - 5 and actual[i] <= predicted[i] + 5:
            #add 1 to the correct iterator
            correct += 1
    #return percentage of predictions that were correct
    return correct / float(len(actual)) * 100.0
 
    
def trainTree(x_train, y_train):
    '''Fit the regression tree with bagging'''
    regressor = RandomForestRegressor(max_depth=20, random_state=10, min_samples_split=2,
    n_estimators=20, bootstrap=True, verbose=0, criterion='mae', n_jobs=-1, max_features='sqrt')
    regressor.fit(x_train,y_train)
    return regressor 

def train_crossValidate(dataset, n_folds):
    '''Trains the trees while cross validating '''

    #folds are the subsamples used to train and validate model
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    Trees = list()
    #for each subsample
    for i,fold in enumerate(folds):
        #create a copy of the data
        train_set = list(folds)
        #remove the given subsample
        train_set.pop(i)
        train_set = sum(train_set, [])
        #init a test set
        test_set = list(fold)
        # construct training data
        xtrain = np.array(train_set)[:,[0,1,2,3,4,5,6]]
        ytrain = np.array(train_set)[:,7]
        xtest  = np.array(test_set)[:,[0,1,2,3,4,5,6]]
        ytest  = np.array(test_set)[:,7]
        #print(ytest)

        # build tree
        trees = trainTree(xtrain, ytrain)

        # compare the accuracy
        train_acc = accuracy_metric(ytrain, trees.predict(xtrain))
        test_acc  = accuracy_metric(ytest, trees.predict(xtest))

        #add it to scores list, for each fold 
        scores.append([train_acc, test_acc])

        #add tree
        Trees.append(trees)

        #return all accuracy scores
    return [scores, Trees]


# Parse the user flags
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--NFold', type=int, help='Number of folds to be used when cross validating')
    parser.add_argument('-s', '--SaveTree', action='store_true', help='Specifies wether the best tree should be saved or not')

    return parser.parse_args(argv)

# Entry point
def main(args):
    if not args.NFold:
        args.NFold = 10
        print("!!!!!!!!  The number of folds should be specified using the flag -n for cross validation. Default value has been set to 10 !!!!!!!!")

    # train the trees and CV
    scores, trees = train_crossValidate(Xtrain, args.NFold)
    #print('Scores: %s' % scores)
    scores = np.array(scores)

    newScores = list()
    #choose the best tree: smallest gap between train and test accuracy towards 100
    for i in range(len(trees)):
        tracc = accuracy_metric(Xtrain[:,7], trees[i].predict(Xtrain[:,[0,1,2,3,4,5,6]]))
        teacc = accuracy_metric(Ytest, trees[i].predict(Xtest))
        print('Train accuracy:', tracc )
        print('Test accuracy:',  teacc )
        print('')
        newScores.append([tracc, teacc])

    newScores_T = 100 - np.array(newScores) 

    # note that some values might be negative. This means the test error is lower than the training error 
    # This should be the best scenario
    best_index = np.argmin(np.abs(newScores_T[:,0] - newScores_T[:,1]))




    #print the best tree accuracy for the whole dataset
    print('Best Train accuracy:', accuracy_metric(Xtrain[:,7], trees[best_index].predict(Xtrain[:,[0,1,2,3,4,5,6]])) )
    print('Best Test accuracy:',  accuracy_metric(Ytest, trees[best_index].predict(Xtest)) )
    print('Mean Train accuracy:', np.array(newScores).mean(axis=1)[0])
    print('Mean Test accuracy:',  np.array(newScores).mean(axis=1)[1] )

    # save the tree for prediction 
    if args.SaveTree :
        filename = 'randomForestTreeModel.dat'
        pickle.dump(trees[best_index], open(filename, 'wb'))
        print('saved model to file!')



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

#print(train_reg_error)
#print(test_reg_error)

