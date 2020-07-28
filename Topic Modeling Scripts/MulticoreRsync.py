#!/bin/env python3.7
#----------DEFINE START TIME------------------
import time
from datetime import date, datetime, timedelta
today = date.today()
run_date = today.strftime("%d.%m.%Y")
start_now = datetime.now()

#------------IMPORT PACKAGES
import os
import pandas as pd
import numpy as np
import re
from htrc_features import feature_reader
from htrc_features import Volume
from htrc_features import utils
import pyLDAvis.gensim
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
wnl = WordNetLemmatizer()

#### added print
print('Packages imported')

#------IMPORT VARIABLES FROM THE INPUT FILE--------
from MulticoreInput import filename, descriptor, htids, workers, num_topics, chunksize, passes, iterations, eval_every
print('variables imported')

#----------DIRECTORY OPERATORS-------------------
cwd = os.getcwd()
today = date.today()
run_date = today.strftime("%d.%m.%Y")
run_dir = ('/Users/bkrien/RunDirectories/'+run_date+'/'+filename)
#create output directory for each run date
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
print('directory created')
os.chdir(run_dir)

#create the Argon Run Report
run_report = f"ArgonRunReport_{filename}_{run_date}.txt"
f = open(run_report, "w+")
print('report created')

#initial run report file write
f.write("ARGON RUN REPORT\n")
f.write(f"Run Name: {filename}\n")
f.write(f"Run Description: {filename}\n")
f.write(f"Run for: {run_date}\n")
f.write(f"Files run: {htids}\n")
f.write("\n PARAMETERS\n")
f.write(f"Number of Topics: {num_topics}\n")
f.write(f"Workers: {workers}\n")
f.write(f"Chunksize: {passes}\n")
f.write(f"Iterations: {iterations}")
#f.write(f"CoherenceScore: {coherence_lda}\n")
print('initial report written')

#---------IMPORT HTRC FILES--------
os.chdir('..')
mag_dir = ('/Users/bkrien/MagFiles/')
os.chdir(mag_dir)

print('beginning htrc import')
full_tokens =np.array([])
mag_names = ''
for htid in htids:
    vol = Volume(htid)
    print(vol.title)

     #convert to tokenlist
    tlist = vol.tokenlist(section='body', case=False)

    #reset the index to make the dataframe easier to manipulate
    tlist_reset = tlist.reset_index()

    #remove nonalphanumeric characters
    def alphanum(row):
        return re.sub('[^a-zA-Z]', '', row)

    tlist_reset.loc[:,'lowercase'] = tlist_reset.loc[:, 'lowercase'].apply (lambda row: alphanum(row))

    #remove the tokens that are less than three characters long
    tlist_len = tlist_reset[tlist_reset.loc[:, 'lowercase'].map(len) >= 3]

    #filter the tokens based on a select part of speech list
    pos = ['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
    #use this if you want to experiment with the different parts of speech
    pos0 = ['UH']

    tlist_pos = tlist_len[tlist_len.loc[:, 'pos'].isin(pos)]

    #remove the stopword from the set of tokens
    stops = set(stopwords.words('english'))
    tlist_stop = tlist_pos[~tlist_pos.loc[:, 'lowercase'].isin(stops)]

    #create a new column with simplified pos tags for the WordNetLemmatizer
    #----------This throws a warning about chained indexing that doesn't come up in the practice file-------
    #---------The method suggested (.loc) doesn't seem to allow the code to run. Should review and fix------
    tlist_pos['pos_new'] = tlist_pos.loc[:, 'pos'].str.extract(r'(^\w{1})')
    convert_dict = {'page': int,
                   'section':object,
                   'lowercase': object,
                   'pos': object,
                   'pos_new': object}
    tlist_pos = tlist_pos.astype(convert_dict)
    tlist_pos.loc[:, 'pos_new'] = tlist_pos.loc[:, 'pos_new'].str.lower()
    tlist_pos.loc[:, 'pos_new'] = tlist_pos.loc[:, 'pos_new'].replace(r'j','a')

    #lemmatize the token using the new pos tag
    def lemma(row):
        lemma = wnl.lemmatize(row['lowercase'], row['pos_new'])
        return lemma

    tlist_pos['lemma'] = tlist_pos.apply (lambda row: lemma(row), axis=1)

    def token_return(row):
        output = ' '
        i = 1
        if row['count'] > 1:
            while i < row['count']:
                output += row['lemma'] + ' '
                i += 1
                return output
    mag_names = mag_names+vol.title+', '

    tlist_pos.loc[:, 'tokens'] = tlist_pos.apply (lambda row: token_return(row), axis = 1)
    tlist_pos.loc[:, 'tokens'] = tlist_pos['tokens'].fillna('') + tlist_pos['lemma']
    tokens = tlist_pos['tokens'].apply(word_tokenize)
    full_tokens = np.concatenate((full_tokens, tokens), axis = None)
#### added print
print('tokenization complete')
#full_tokens.to_csv('tokens.csv')

#-------CREATE DICTIONARY & CORPUS------
#create dictionary
tokens = full_tokens
os.chdir('..')
os.chdir(run_dir)
from gensim import corpora
dictionary=corpora.Dictionary(tokens)
dictionary.save('dictionary')
print('dictionary created')

#creation corpus
corpus = [dictionary.doc2bow(token) for token in tokens]
corpora.MmCorpus.serialize('corpus.mm', corpus)
print('corpus complete')

#-------LDA IMPLEMENTATION-----------
import gensim
from gensim import models, corpora, utils, parsing, similarities
from gensim.models.ldamulticore import LdaMulticore

#Make an index to word dictionary
temp = dictionary[0] #This is only to load the dictionary
id2word = dictionary.id2token

lda = LdaMulticore(corpus,
    id2word=id2word,
    workers=workers,
    num_topics=num_topics,
    chunksize=chunksize,
    passes=passes,
    iterations=iterations)

lda.save('lda_model.gensim')
lda_model = 'lda_model.gensim'
print('lda model created')

#This prints the topics with their relative weights to a .txt file
topics = lda.print_topics(num_words = 20)
topic_file = f"20topics_{filename}_{run_date}.txt"
f = open(topic_file, "w+")
for topic in topics:
    f.write(str(topic))
topics = lda.print_topics(num_words = 7)
topic_file = f"7topics_{filename}_{run_date}.txt"
f = open(topic_file, "w+")
for topic in topics:
    f.write(str(topic))
print('topic models created')

#This exports the topics and their relative weights to a csv file
top_words_per_topic = []
for t in range(lda.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda.show_topic(t, topn = 20)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")

#---------COHERENCE SCORE----------------------
#Can add in coherence score later....
#....
#...
#..
#.

#--------CREATE VISUALIZATION-------------
lda = gensim.models.ldamodel.LdaModel.load(lda_model)

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_display, f"viz_{filename}_{run_date}.html")

print('data viz created')

#---------FINAL REPORT WRITING--------------------
####Moving the report writing up and making this portion of the code dormant
#### until I can figure out what to do with it to make sure that this section
#### of the report is being written correctly.
#define end time
#end_now = datetime.now()
#start_time = start_now.strftime("%H.%M.%S")
#end_time = end_now.strftime("%H.%M.%S")
#duraction = end_now - start_now

#report writing
#f.write(f"Argon run begun at {start_time}\n")
#f.write(f"Completed on: {end_time}\n")
#f.write(f"Run Time: {duration}\n")
