"""

Author:  Ozwald Cavendish
Date:  05/09/2019

Scripting for running my news-scraping helper functions for me, dumping the results to some json files
in Dropbox.
"""

import sys
import os
import json
import time

import pandas as pd

from datetime import datetime

from lib.pumps import *


def run_rss(settings):
    
    try:
        # Firstly, load RSS feed list
        feeds_df = pd.read_csv(settings['rss_feed_list'])
        
        rss_corpus = []
        for index, row in feeds_df.iterrows():
            rss_corpus = rss_corpus + get_rss_items(row['url'])
    
        # Dump the corpus to file, record the date and time in the filename
        filename = settings['output_folder'] + "/RSS_corpus_{}.json".format(datetime.now().strftime("%Y-%m-%d %H%M").replace(" ", "_") )
        
        with open(filename, "w") as f:
            json.dump(rss_corpus, f)
        return 0
    
    except Exception as e:
        print(e)
        return -1
    

def run_bing(settings):
    
    try:
        # Firstly, load search terms
        searches = settings['search_list']
        
        search_corpus = []
        for term in searches:
            # Replace operation to translate quoted quotes from single (needed for json) to double (needed for bing API)
            # Yes, this is a stupid problem
            search_corpus = search_corpus + get_bing_items(term.replace("'", '"'), settings, filter_uk=False)
            
            # Pause two seconds to avoid surpassing the rate limit on the API
            time.sleep(2)
        
        # Dump the corpus to file, record the date and time in the filename
        filename = settings['output_folder'] + "/bing_corpus_{}.json".format(datetime.now().strftime("%Y-%m-%d %H%M").replace(" ", "_"))
        
        with open(filename, "w") as f:
            json.dump(search_corpus, f)
        return 0
    
    except Exception as e:
        print(e)
        return -1


def run_bing_disasters(settings):
    
    try:
        # Firstly, load search terms
        searches = settings['disaster_search_list']
        
        search_corpus = []
        for term in searches:
            # Replace operation to translate quoted quotes from single (needed for json) to double (needed for bing API)
            # Yes, this is a stupid problem
            search_corpus = search_corpus + get_bing_items(term.replace("'", '"'), settings, filter_uk=False)
            
            # Pause two seconds to avoid surpassing the rate limit on the API
            time.sleep(2)
        
        # Dump the corpus to file, record the date and time in the filename
        filename = settings['output_folder'] + "/bing_disaster_corpus_{}.json".format(datetime.now().strftime("%Y-%m-%d %H%M").replace(" ", "_"))
        
        with open(filename, "w") as f:
            json.dump(search_corpus, f)
        return 0
    
    except Exception as e:
        print(e)
        return -1
        

def run_all(settings_filepath):
    
    with open(settings_filepath, "r") as f:
        settings = json.load(f)
    
    print("Calling Bing API")
    run_bing(settings)
    
    print("Calling Bing API for disasters")
    run_bing_disasters(settings)
    
    print("Running RSS scraper")
    run_rss(settings)
    
    return 0


if __name__ == "__main__":
    run_all(sys.argv[1])
