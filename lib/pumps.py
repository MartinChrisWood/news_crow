"""
Utility functions for scraping large numbers of RSS feeds, handles extracting info from multiple feeds, cleaning
and returning that data in a DB ready schema/format for me.
"""

import requests
import feedparser

from datetime import datetime
from urllib.parse import urlparse


def get_rss_items(url):
    """ Pumps a single RSS feed, connecting, downloading and parsing the text from it. """
    feed_object = feedparser.parse(url)
    
    corpus = []
    for article in feed_object['entries']:
        try:
            corpus.append({"title": article['title'],                    # News titles
                           "summary": article['summary'].split("<")[0],  # payload (sans any HTML stuff)
                           "date": article['published'],
                           "link": article['links'][0]['href'],          # associated links
                           "source_url": url,
                           "retrieval_timestamp": str(datetime.now()),
                           'origin': "rss_feed",
                           'raw': article})

        except KeyError as e:
            print("failed on ", article, e)
            continue
    
    return corpus


def get_bing_items(search_term, settings, filter_uk=True):
    """ Calls the Bing news API with a search term, downloading and parsing the text from it. """
    # Metadata and verification etc. for query
    headers = {"Ocp-Apim-Subscription-Key": settings['cognitive_resources']["key"]}
    parameters = settings['news_api_parameters']
    parameters['q'] = search_term
    search_url = settings['cognitive_resources']['endpoint'] + settings['cognitive_resources']['news_api_endpoint']
    
    # Call the API
    response = requests.get(search_url, headers=headers, params=parameters)
    response.raise_for_status()
    search_results = response.json()
    
    # Extract articles, ditch metadata, filter to .co.uk sites if desired.
    if filter_uk:
        articles = [article for article in search_results['value'] if ".co.uk" in article['url']]
    else:
        articles = [article for article in search_results['value']]
    
    # Rename/filter data to my own schema
    corpus = []
    for article in articles:
        try:
            clean = {'title': article['name'],
                     'summary': article['description'],
                     'date': article['datePublished'],
                     'link': article['url'],
                     'source_url': urlparse(article['url']).netloc,
                     'retrieval_timestamp': str(datetime.now()),
                     'origin': "bing_news_api",
                     'raw': article}
            corpus.append(clean)
        
        except KeyError as e:
            print("failed on ", article, e)
            continue
    
    return corpus
    
