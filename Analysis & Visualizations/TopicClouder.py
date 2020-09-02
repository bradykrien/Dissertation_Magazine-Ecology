import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_file',
                        help='csv file output from gensim topic modeling, arranged in columns')
    args = parser.parse_args()
    if not os.path.exists('Visualizations'):
        os.makedirs('Visualizations')
    df = pd.read_csv(args.topic_file, index_col=0)
    df_multi = df.set_index(['Topic'])
    topics = []
    topics = df.Topic.unique()
    tup_dict = {}
    for topic in topics:
        top_df = df_multi.loc[(topic),]
        tuples = [tuple(x) for x in top_df.to_numpy()]
        tup_dict[topic] = tuples
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
