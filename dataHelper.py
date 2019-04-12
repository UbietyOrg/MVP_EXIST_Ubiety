# This file define the possible features of the data set.

import numpy as np
import random as rd


# Header - Features description
Header = ['Type_of_event', 'Distance', 'Warranty', 'Participants', 'Preference', 'Profession', 'Age', 'show_up_Rate']

# Type of event - categorical
Toe_c = np.array(['Exhibition', 'Workshop', 'Summit', 'Congress', 'Others' ])
# Type of event
Toe = np.array([7, 8, 9, 10, 11])

#Distance. Between the even location and the user location
Distance = np.array([0, 10, 20, 50, 100, 200, 300, 500, 800, 1000])

# Waranty sum 
Warranty = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, 10000])

# Amount of participants for event
Participants = np.array([1, 5, 10, 20, 50, 200, 1000, 10000])

# User show up rate 
U_ShowUp = np.array([i*5 for i in range(0,21)])

# User event preference - categorical
EventPref_c = np.array(['Exhibition', 'Workshop', 'Summit', 'Congress', 'Others'])
# User event preference 
EventPref = np.array([7, 8, 9, 10, 11])

# User's profession - categorical
Profession_c = np.array(['Student', 'Employee', 'Employer', 'Others'])
# User's profession
Profession = np.array([1, 2, 3, 4])

# The participant's age is drawn for simplicity purposes from an gaussian distribution with
# with mean 35 years and variance 30 years 
# A much accurate probability distribution will look as follow
#
#              * * *   
#           *          *  
#         *                 *
#        *                        *
#       *                                   *           *
#  --*---------------------------------------------------------> Years
#  0  10   20   30  40  50   60   70        80 

# gaussian distribution with 1000 values
# take the absolute value as the normal distribution extends from -inf to + inf
Age = np.abs( np.random.normal(35, 30, 1000) )

def getRandomAge():

    age = int(Age[rd.randint(0, len(Age) -1)])
    if age > 90:
        getRandomAge()

    return age

def getToeCategory(val):
    ''' Returns the type of event category according to the given value'''
    # remove threshold
    val -=7
    return Toe_c[val]

def getToeVal(cat):
    ''' Returns the type of event value according to the given category'''
    # get index from original categorycal list
    index = np.argwhere(Toe_c == cat)

    if index.size ==0:
        index = 2
    else: 
        index = np.squeeze(index[0])

    return Toe[index]

def getEventPrefCategory(val):
    ''' Returns the user event's preference category according to the given value'''
    # remove threshold
    val -=7
    return EventPref_c[val]

def getEventPrefVal(cat):
    ''' Returns the user event's preference value according to the given category'''
    # get index from original categorycal list
    index = np.argwhere(EventPref_c == cat)

    if index.size ==0:
        index = 2
    else: 
        index = np.squeeze(index[0])

    return EventPref[index]

def getProfessionCategory(val):
    ''' Returns the user's profession category according to the given value'''
    val -=1
    return Profession[val]

def getProfessionVal(cat):
    ''' Returns the user's profession value according to the given category'''
    # get index from original categorycal list
    index = np.argwhere(Profession_c == cat)

    if index.size ==0:
        index = 2
    else: 
        index = np.squeeze(index[0])

    return Profession[index]


