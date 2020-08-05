"""
Martin Wood - 14/07/2019
For convenience; various functions that the experiment notebooks would be cleaner without
"""

import os
import re
import json
import gensim
import random

import numpy as np
import pandas as pd

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser

from nltk.stem.porter import *

# Define which stemmer to use in the pipeline later
stemmer = PorterStemmer()

# Spacy is used for POS tagging, because it has awesome neural network shizzle
import spacy
nlp = spacy.load('en_core_web_sm')

# Useful flatten function from Alex Martelli on https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
flatten = lambda l: [item for sublist in l for item in sublist]


def clean_text(article_text, brutal=False):
    """
    Utility function for cleaning up text for me.
    There's probably better ways to prepare data.
    I have a rapidly growing pile of annoying exceptions here,
    should probably go learn about text encodings!
    """
    article_text = re.sub(r'<b>|</b>|[&#39]', '', article_text)     # Remove annoying tags
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)         # Gets rid of numbers
    article_text = re.sub(r'\s+', ' ', article_text)                # Replaces all forms of white space with single space
    article_text = re.sub(r'&apos;', '', article_text)              # Stupid apostrophe marker, I don't know how I ended up saving that
    article_text = re.sub(r'apos;', '', article_text)              # Stupid apostrophe marker, I don't know how I ended up saving that
    article_text = re.sub(r'8217;', '', article_text)               # Special char that slips through
    article_text = re.sub(r'8220;', '', article_text)               # Special char that slips through
    article_text = re.sub(r'8221;', '', article_text)               # Special char that slips through
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


def corpus_world_loader(directory, corpus_tag, drop_raw=True):
    """
    For loading my corpus files from RSS feeds specifically,
    Filters those that are not from a world news site
    """
    # Firstly, load RSS feed list
    feeds_df = pd.read_csv("D:/Dropbox/news_crow/rss_urls.csv")
    
    world_urls = list(feeds_df[feeds_df['type']=='world']['url'])
    
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
                if article['source_url'] in world_urls:
                    # Optional, don't bother loading up the original raw response (saves memory)
                    if drop_raw:
                        article.pop("raw")
                        
                    compendium.append(article)
                
    return pd.DataFrame(compendium)


def load_clean_world_corpus(directory, corpus_tag, drop_raw=True, brutal=False):
    """ All common pre-processing. """
    corpus = corpus_world_loader(directory, corpus_tag, drop_raw=drop_raw)

    # Filter to only the .uk vendors
    corpus = corpus[corpus['link'].str.contains(".uk/")]

    # Drop duplicates based on actual text
    corpus = corpus.drop_duplicates("summary")

    # Clean whatever's survived
    corpus['clean_text'] = corpus[['title', 'summary']].apply(lambda x: clean_text('.  '.join(x)), axis=1)

    return corpus


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


def get_phrased_nouns(sentences):
    """ Use spacy to get all of the actual entities, conjoin bigram nouns. """

    # Get the lists of nouns
    noun_lists = []
    for doc in sentences:
        parsed = nlp(doc)
        
        # Reduce to words that are proper nouns
        noun_string = " ".join([token.text for token in parsed if token.pos_ == 'PROPN'])
        
        # Apply cleaning and remove any strings left empty
        noun_lists.append([x.strip() for x in preprocess_description(noun_string) if len(x.strip()) != 0])

    # Build the phrase model
    phrases = Phrases(noun_lists, min_count=5, threshold=0.5)

    # Get the set of phrases present in the model
    results = []
    for nouns in noun_lists:
        results.append(phrases[nouns])

    return results


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
    
    
