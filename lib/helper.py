"""
Martin Wood - 14/07/2019
For convenience; various functions that the experiment notebooks would be cleaner without
"""


import re
import os
import json

import pandas as pd

from datetime import datetime as dt


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
