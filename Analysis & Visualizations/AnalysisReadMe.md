## Analysis & Visualization ReadMe

This directory contains code for the visualizations taht I developed along with some sample visualizations and data that generated through the course of this dissertation project. 

### Visualizations
__TopicClouder__
This visualization file is a python script that can be called on a csv of topics exported from gensim. The csv should be in a three colum format with "Topic" "Word" and "P" (word frequency/rate) as the columns. Calling this script on a topic file will create a visualizations subdirectory within the current working directory and generate a png wordcloud for each of the topics within this directory. The only set parameters for these wordclouds is that they have a whitebackground and are sized to 600x400 pixels. Additional parameters (scale, font size, color, word limits, etc.) can be added into the script based on the parameters for wordclouds found here: https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html

### Sample Data
It also includes a variety of sample data to give a sense of the kinds of output and topics that are developed within some of the subdirectories. The subdirectories are named based on the magazine or magazines that are modeled with the number of topics modeled (e.g. NatGeo20 is a modeling of 20 topics in a set of *National Geographic* magazines while Total25 contains the results from all of the magazines in the datasets modeled for 25 topics). Each subdirectory contains: 
1. The processed dataset
1. A run report with the hyperparameters for the topic, a brief description of the model, and the HathiTrust IDs for the volumes in the corpus
1. A list of the top 20 words within each topic and their relative weight
1. An interactive HTML visualization built with the pyLDAvis library
1. The gensim corpus file, stored in a .mm format
1. The token index
1. A numpy array of the topic model

These files can be used to explore or replicate the results.  
