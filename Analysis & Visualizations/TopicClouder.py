# This is a function that can be called on a csv file of topics generated from
#gensim
# The csv file of topics should have three columns: Topic, Word, and P (frequency)
# When you call this funtion on the csv file, it will create a "Visualizations"
# directory in the current working directory and create a visualization for each
# topic as a png fill within that directory

# Import packages
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import argparse

# Define callable function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_file',
                        help='csv file output from gensim topic modeling, arranged in columns')
    args = parser.parse_args()
    #Create directory if it doesn't already exist
    if not os.path.exists('Visualizations'):
        os.makedirs('Visualizations')
    #convert csv into a pandas dataframe
    df = pd.read_csv(args.topic_file, index_col=0)
    df_multi = df.set_index(['Topic'])
    topics = []
    #extract a list of topics (will be an array of numbers)
    topics = df.Topic.unique()
    #create a dictionary of tuple arrays with topics as the keys
    tup_dict = {}
    for topic in topics:
        top_df = df_multi.loc[(topic),]
        tuples = [tuple(x) for x in top_df.to_numpy()]
        tup_dict[topic] = tuples
    #generate a wordcloud for each topic and export it to Vis directory
    wc = WordCloud(background_color='white', width=600, height=300)
    for topic, tuples in tup_dict.items():
        name = 'wordcloud_topic{}.png'.format(topic)
        wordcloud = wc.generate_from_frequencies(dict(tuples))
        plt.figure
        plt.imshow(wc)
        plt.axis("off")
        wordcloud.to_file(os.path.join('Visualizations', name))

if __name__ == '__main__':
    main()
