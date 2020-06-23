"""
Martin Wood - 14/07/2019
For convenience; various functions that the experiment notebooks would be cleaner without
"""

import os
import re
import json
import gensim

import numpy as np
import pandas as pd
import seaborn as sns

from datetime import datetime as dt
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem.porter import *

# Define which stemmer to use in the pipeline later
stemmer = PorterStemmer()

# Useful flatten function from Alex Martelli on https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
flatten = lambda l: [item for sublist in l for item in sublist]


def clean_text(article_text, brutal=False):
    """ Utility function for cleaning up text for me.  There's probably better ways to prepare data. """
    article_text = re.sub(r'<b>|</b>|[&#39]', '', article_text)     # Remove annoying tags
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)         # Gets rid of numbers
    article_text = re.sub(r'\s+', ' ', article_text)                # Replaces all forms of white space with single space
    if brutal:                                                      # Optional, all non alpha-numeric characters removed
        article_text = re.sub('r[^0-9A-Za-z ]', "", article_text)
    return(article_text)


def corpus_loader(directory, corpus_tag, drop_raw=True):
    """ For loading my corpus files """

    # Get a list of all corpus files
    files = [x for x in os.listdir(directory) if x.endswith(".json") and ("corpus" in x) and (corpus_tag in x)]

    # Implement filters here if I ever feel I need them
    # Filter by dates if needed
    #datetimestamps = [dt.strptime(":".join(re.findall(r"[0-9-]{1,}", x)), "%Y-%m-%d:%H%M")  for x in files]

    # Iterate through and load up every file in sequence
    compendium = []

    total = len(files)
    print("Total files: {}".format(total))

    i = 0
    for filename in files:

        # Remark on progress
        i += 1
        if (i % (int(total / 10)) == 0):
            print("%.1f percent of files read." % (100.0 * i / total))

        with open(directory + "/" + filename, "r") as f:
            articles = json.load(f)
            for article in articles:

                # Optional, don't bother loading up the original raw response (saves memory)
                if drop_raw:
                    article.pop("raw")
                compendium.append(article)

    return pd.DataFrame(compendium)


def load_clean_corpus(directory, corpus_tag, drop_raw=True, brutal=False):
    """ All common pre-processing. """
    corpus = corpus_loader(directory, corpus_tag, drop_raw=drop_raw)

    # Filter to only the .uk vendors
    corpus = corpus[corpus['link'].str.contains(".uk/")]

    # Drop duplicates based on actual text
    corpus = corpus.drop_duplicates("summary")

    # Clean whatever's survived
    corpus['clean_text'] = corpus[['title', 'summary']].apply(lambda x: clean_text('.  '.join(x)), axis=1)

    return corpus


def preprocess_description(description):
    """ Helper, tokeniser """
    return( [stemmer.stem(token) for token in simple_preprocess(str(description)) if token not in STOPWORDS] )


def get_stats(df):
    """ Helper, for printing basic info on corpus extent """
    
    df['doc_size'] = df['clean_text'].apply(lambda x: len(x.split()))

    print(np.mean(df['doc_size']))
    print(df.shape[0])
    print(max(df['date']))
    print(min(df['date']))
    
    return 0


def get_keyword_stats(df, search_term_path = "D:/Dropbox/news_crow/scrape_settings.json"):
    """Retrive the set of search terms used for Bing, sum stories that contain them """
    with open(search_term_path, "r") as f:
        scrape_config = json.load(f)
        
    search_terms = scrape_config['disaster_search_list']
    search_terms = re.sub(r"[^0-9A-Za-z ]", "", " ".join(search_terms)).lower().split()
    search_terms = set(search_terms)
    
    term_results = {}
    
    for term in search_terms:
        term_results[term] = sum(df['clean_text'].apply(lambda x: term in x.lower()))
    
    return(term_results)
    

def get_corpus_model_coherence(df, cluster_column="cluster", tokens_column="tokens"):
    """ 
    Encapsulates entire coherence model-building process for (flat) models
    """
    # Create the vocabulary record
    bow_dictionary = gensim.corpora.Dictionary(list(df[tokens_column]))
    
    # Create a BOW model
    bow_corpus = [bow_dictionary.doc2bow(doc) for doc in df[tokens_column]]
    
    # Flattened list of all tokens for all documents for each "topic"
    topics = {}
    topics_lengths = {}
    
    for topic in pd.unique(df['cluster']):
        subset = df[df['cluster'] == topic]
        
        topics_lengths[topic] = subset.shape[0]
        
        topics[topic] = flatten(list(subset['tokens']))
    
    # Calculate ALL THE COHERENCE
    coherence_models = {}
    
    # c_v is most performant indirect confirmation measure
    cm1 = CoherenceModel(topics=list(topics.values()),
                         texts=list(df['tokens']),
                         dictionary=bow_dictionary,
                         coherence='c_v')
    coherence_models['c_v'] = cm1
    
    # c_npmi is most performant direct confirmation measure (that I don't have to implement myself)
    cm2 = CoherenceModel(topics=list(topics.values()),
                         texts=list(df['tokens']),
                         dictionary=bow_dictionary,
                         coherence='c_npmi')
    coherence_models['c_npmi'] = cm2
    
    return(coherence_models, topics_lengths)


def report_corpus_model_coherence(df, cluster_column="cluster", tokens_column="tokens"):
    """
    Creates two key coherence models (C_v, NPMI) and reports coherences,
    plus coherence/topic distribution
    """
    coherence_models, topics_lengths = get_corpus_model_coherence(df,
                                                                  cluster_column="cluster",
                                                                  tokens_column="tokens")
                                                                  
    for key in coherence_models.keys():
        
        # Extract the model scores and sizes
        topic_features = pd.DataFrame({"topic_coherence": list(coherence_models[key].get_coherence_per_topic()),
                                       "topic_sizes": list(topics_lengths.values())})
    
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    topic_features['topic_sizes'].hist(ax=axs[0])
    topic_features['topic_coherence'].hist(ax=axs[1])
    sns.scatterplot(x='topic_sizes', y='topic_coherence', data=topic_features, ax=axs[2])

    return coherence_models
    
    
def load_evaluate_corpus(data_path):
    """ Helper, process a corpus csv, return its coherence scores """
    df = pd.read_csv(data_path)
    
    df["tokens"] = df["clean_text"].apply(preprocess_desc)
    
    print("Number of topics is {}".format(len(pd.unique(df['cluster']))))
    
    return get_corpus_model_coherence(df)
    