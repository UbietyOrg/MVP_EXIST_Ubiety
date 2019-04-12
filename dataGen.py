#### This file generates artificial data for a random forest model to be developped as MVP ####

## Each single instance of the data represents a participant.. 
## This instance could be repeated across the dataset as participants might have the same behaviours. The participant s are not sorted according to any given event. 
## Plausibility of data is made by assuming the features are drawn from multiple distributions 


# Dependencies 
import numpy as np
import random as rd
import os 
import argparse
import sys
import dataHelper
import math


def log_data_element(elm):
    '''Append a line in a file '''
    with open("simple_data.csv","a+") as f:
        f.write(elm + str('\n'))

def getDistance_P(dist):
    '''Models the show up probability according to the distance between the user and the event.
    This is done using a modified sigmoid function'''

    def sigmoid_d(x):
        '''Appraximate the show up probability -  Distance'''
        return  1 -  1 / (1 + math.exp(-0.8*(x- 6) ) )

    # As the distance elements are discret and sorted, use the  array index for computing the probs
    dist_index = int(np.argwhere(dataHelper.Distance == dist)[0])
    #print('Distance:',sigmoid_d(dist_index))
    return sigmoid_d(dist_index)

def getWarranty_P(war):
    '''Models the show up probability according to the warranty sum to deposit.
    This is done using a modified sigmoid function'''

    def sigmoid_w(x):
        '''Appraximate the show up probability - Warranty sum'''
        return  1 - 1 / ( 1 +  math.exp( -(1.5*x - 8) ) )

    # As the distance elements are discret and sorted, use the  array index for computing the probs
    war_index = int(np.argwhere(dataHelper.Warranty == war)[0])
    #print('Warranty:',sigmoid_w(war_index))
    return sigmoid_w(war_index)


def getParticipants_P(par):
    '''Models the show up probability according to the event attendance.
    This is done using a modified sigmoid function'''

    def sigmoid_p(x):
        '''Appraximate the show up probability - participants'''
        return  1 -  1 / ( 1.4 + math.exp( -( 2*x - 8) ) )

    # As the distance elements are discret and sorted, use the  array index for computing the probs
    par_index = int(np.argwhere(dataHelper.Participants == par)[0])
    #print('participants:',sigmoid_p(par_index))
    return sigmoid_p(par_index)

def getProfession_P(prof):
    '''Approximate the show up probability acordint to the participant's profession '''
    if prof == dataHelper.Profession[0]:    # student
        return 0.5
    elif prof == dataHelper.Profession[1]:  # employee
        return 0.8
    elif prof == dataHelper.Profession[2]:  # employer
        return 1
    elif prof == dataHelper.Profession[3]:  # others
        return 0.7
    else:
        return 0.7

def getPreference_P(toe, pref):
    ''' Approximates the participant event's preference probability '''
    if toe == pref:
        return 1
    else:
        return 0.6
    

# Create the dataset
def create_data(args):

    # construct header string
    head = ''
    for h in dataHelper.Header:
        head += str(h) + str(',')
    
    # remove trailing ','
    head = head[:-1]
    #print(head)
    log_data_element(head)

    # default typeof event
    toe = dataHelper.Toe[0]

    # generate the dataset for only a random type of event
    if args.single:
        toe = dataHelper.Toe[ rd.randint(0,len(dataHelper.Toe) -1)]

    for _ in range(5000):
        # generate the dataset for only a random type of event
        if args.all:
            toe = dataHelper.Toe[ rd.randint(0,len(dataHelper.Toe) -1)]

        d    = dataHelper.Distance[ rd.randint(0,len(dataHelper.Distance) -1)]
        w    = dataHelper.Warranty[ rd.randint(0,len(dataHelper.Warranty) -1)]
        p    = dataHelper.Participants[ rd.randint(0,len(dataHelper.Participants) -1)]
        ep   = dataHelper.EventPref[ rd.randint(0,len(dataHelper.EventPref) -1)]
        prof = dataHelper.Profession[ rd.randint(0,len(dataHelper.Profession) -1)]
        age  = dataHelper.getRandomAge()

        # compute the participant show probability by weighting other probabilities
        # continiuous probability 
        usp = 25*getDistance_P(d) + 25*getWarranty_P(w) + 25*getParticipants_P(p) + \
             15*getProfession_P(prof) + 10*getPreference_P(toe,ep)

        # discretise the show up probability according to our predefinitions 
        # get next discretised value from show up array
        #print('Continiuos show up:', usp)
        usp = dataHelper.U_ShowUp[ int(np.argwhere(dataHelper.U_ShowUp >= usp )[0]) ]
        #print('Discrete show up:', usp)

        # construct a feature 
        elm = str(toe) + str(',') + str(d) + str(',') + str(w) + str(',') + str(p) + str(',') + \
              str(ep) + str(',') + str(prof) + str(',') + str(age) + str(',')  + str(usp) 

        # write feature to file 
        log_data_element(elm)

# Parse the user flags
def parse_arguments(argv):
  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--single', action='store_true', help='Generate the dataset for only one event category')
  parser.add_argument('-a', '--all', action='store_true',help='Generate the dataset for or all event categories')

  return parser.parse_args(argv)

# Entry point
def main(args):

    # delete old dataset 
    if os.path.exists("simple_data.csv"):
        print('Removing old data file ...')
        os.remove("simple_data.csv")
        print('Removed old data file')
    
    print('creating new data ...')
    create_data(args)  
    print('Done creating new data')

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

