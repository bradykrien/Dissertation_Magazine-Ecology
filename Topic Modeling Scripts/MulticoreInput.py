#!/bin/env python3.7
#This file contains the variables for the Multicore implementation
#This includes the lists of specific files that will be used in the modeling

#create a file name that will be combined with the date in naming conventions
filename = ''

#create a brief description that provides an overview of the run that's
#being undertaken
descriptor = ''


#select the files that you will be using, identifying them by their Hathi Trust # ID
#For example 'ucl.311175023709325',
htids = ['uc1.31175023709325',
             'uc1.31175023709333',
             'uc1.31175023709341',
             'uc1.31175023709358']

#Set training parameters
num_topics = 10
workers = 3
chunksize = 2000
passes = 20
iterations = 400
eval_every = None #Don't evaluate model perplexity, takes too much time
