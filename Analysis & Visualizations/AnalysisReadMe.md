## Analysis & Visualization ReadMe

This directory contains sample visualizations and data that generated through the course of this dissertation project. There are a variety of sample data to give a sense of the kinds of output and topics that are developed. The subdirectories are named based on the magazine or magazines that are modeled with the number of topics modeled (e.g. NatGeo20 is a modeling of 20 topics in a set of *National Geographic* magazines while Total25 contains the results from all of the magazines in the datasets modeled for 25 topics). Each subdirectory contains: 
1. The processed dataset
1. A run report with the hyperparameters for the topic, a brief description of the model, and the HathiTrust IDs for the volumes in the corpus
1. A list of the top 20 words within each topic and their relative weight
1. An interactive HTML visualization built with the pyLDAvis library
1. The gensim corpus file, stored in a .mm format
1. The token index
1. A numpy array of the topic model

These files can be used to explore or replicate the results.  
