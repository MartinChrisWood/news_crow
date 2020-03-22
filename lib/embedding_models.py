"""

Martin Wood - 14/07/2019

For convenience; wraps up a lot of the paraphenalia involved in
getting word/sentence embeddings using various pre-trained models.

"""

import torch
import pickle
import nltk
import spacy

import numpy as np
import pandas as pd

from multiprocessing import Pool
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser

# Local import, Requires local copy of InferSent model code with base
# word2vec models and such copied to correct locations
from lib.InferSent.models import InferSent

# And spacy's parsing needs a trained model loaded in to it
nlp = spacy.load('en_core_web_sm')

# Useful func, return tuple of index and lemmatized proper nouns
def pool_lambda(x):
	return (x[0], [token.lemma_ for token in nlp(x[1]) if token.pos_ == 'PROPN'])

class InferSentModel():
    """
    Encapsulates the entire setup process and default configuration for loading a pre-trained InferSent document
    embeddings model and calculating the embeddings for a given corpus.
    """
    # For InferSent sentence level encoder
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0}

    def __init__(self,
                 sentences,
                 labels,
                 MODEL_PATH = './lib/InferSent/encoder/infersent2.pkl',
                 W2V_PATH = './lib/InferSent/dataset/fastText/crawl-300d-2M.vec',
                 params_model = {}):

        self.MODEL_PATH = MODEL_PATH
        self.W2V_PATH = W2V_PATH

        for key in params_model.keys():
            self.params_model[key] = params_model[key]

        # Save the sentence labels
        self.labels = labels

        # Configure the actual model
        self.model = InferSent(self.params_model)
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model.set_w2v_path(self.W2V_PATH)
        self.model.build_vocab(sentences, tokenize=True)

        self.core_embeddings = self.model.encode(sentences, tokenize=True)


    def get_embeddings(self, labels=True):
        """
        Convenience function for getting the embeddings as an array or
        with the labels.
        """
        if labels:
            return {x[0]:x[1] for x in zip(self.labels, self.core_embeddings)}
        else:
            return self.core_embeddings


    def get_more_embeddings(self, new_sentences, new_labels=None, labels=True):
        """
        Get embeddings for sentences not in the original set. These are
        not stored in the object, merely returned.
        """
        if labels:
            return {x[0]:x[1] for x in zip(new_labels, self.model.encode(new_sentences, tokenize=True))}
        else:
            return self.model.encode(new_sentences, tokenize=True)


class GloveWordModel():
    """
    Encapsulates load and setup process for GloVE word embedding model with summing of vectors over text.
	TODO: extract length of embeddings programmatically from model path, use as class var
    """

    def __init__(self, sentences, labels, MODEL_PATH = "./lib/Glove/glove.6B.50d.txt"):

        # Load the word-vector lookup table
        self.word_embeddings = {}
        with open(MODEL_PATH, encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.word_embeddings[word] = coefs

        self.labels = labels
        nltk.download('stopwords')

        self.core_embeddings = self.get_summed_word_vectors(sentences)


    def clean_sentence(self, sentence, remove_stopwords=True):
        """ Utility, clean brutally. """
        sentence = sentence.replace('[^a-zA-Z]', ' ').lower()

        if remove_stopwords:
            sentence = " ".join([word for word in sentence.split() if word not in stopwords.words('english')])

        return sentence


    def get_summed_word_vectors(self, sentences):
        """ Creates averaged word vectors for each sentence. """
        embeddings = []

        for s in sentences:
            if len(s) != 0:
                cleaned = self.clean_sentence(s)
                v = sum([self.word_embeddings.get(w, np.zeros((50,))) for w in cleaned.split()]) / ( len(cleaned.split()) + 0.001 )

            else:
                v = np.zeros((50, 0))
            embeddings.append(v)

        return np.asarray(embeddings)


    def get_embeddings(self, labels=True):
        """
        Convenience function for getting the embeddings as an array or
        with the labels.
        """
        if labels:
            return {x[0]:x[1] for x in zip(self.labels, self.core_embeddings)}
        else:
            return self.core_embeddings


    def get_more_embeddings(self, new_sentences, new_labels=None, labels=True):
        """
        Get embeddings for sentences not in the original set. These are
        not stored in the object, merely returned.
        """
        if labels:
            return {x[0]:x[1] for x in zip(new_labels, self.get_summed_word_vectors(new_sentences))}
        else:
            return self.model.encode(new_sentences, tokenize=True)


class NounAdjacencyModel():
    """ Models of documents are one-hot-encoded named entity presence or absence. """

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.noun_sets = self.get_phrased_nouns_parallel(self.sentences)

        self.all_nouns = self.get_all_nouns()

        self.entities = self.get_entities(self.noun_sets)

        self.table = pd.DataFrame(data=self.entities, index=self.sentences, columns=self.all_nouns)


    def get_proper_nouns(self, sentences):
        """ Use spacy to get all of the actual entities """
        results = []
        for doc in sentences:
            parsed = nlp(doc)
            results.append(set([token.lemma_ for token in parsed if token.pos_ == 'PROPN']))

        return results


    def get_phrased_nouns_parallel(self, sentences):
        """ Use spacy to get all of the actual entities, conjoin bigram nouns. """

        # I have to take some special measures to preserve ordering
        sent_tups = [(i, sentences[i]) for i in range(len(sentences))]

        # Get the noun lists
        p = Pool()
        noun_tups = p.map(pool_lambda, sent_tups)
        noun_dict = {tup[0]: tup[1] for tup in noun_tups}

        # Build the phrase model
        phrases = Phrases(noun_dict.values(), min_count=5, threshold=0.5)

        # Get the set of phrases present in the model
        results = []
        for i in range(len(sentences)):
            results.append(set(phrases[noun_dict[i]]))

        return results


    def get_phrased_nouns(self, sentences):
        """ Use spacy to get all of the actual entities, conjoin bigram nouns. """

        # Get the lists of nouns
        noun_lists = []
        for doc in sentences:
            parsed = nlp(doc)
            noun_lists.append([token.lemma_ for token in parsed if token.pos_ == 'PROPN'])

        # Build the phrase model
        phrases = Phrases(noun_lists, min_count=5, threshold=0.5)

        # Get the set of phrases present in the model
        results = []
        for nouns in noun_lists:
            results.append(set(phrases[nouns]))

        return results


    def get_all_nouns(self):
        """ Get a set of all detected nouns. """
        all_nouns = set()
        for doc_set in self.noun_sets:
            all_nouns = all_nouns.union(set(doc_set))

        return list(all_nouns)


    def get_entities(self, noun_sets):
        """ Create a table of the nouns' presence or absence in each document. """
        results = []
        for doc in noun_sets:
            results.append( np.asarray([int(x in doc) for x in self.all_nouns]) )

        return np.asarray(results)


    def get_embeddings(self, labels=True):
        """
        Convenience function for getting the embeddings as an array or
        with the labels.
        """
        if labels:
            return {x[0]:x[1] for x in zip(self.labels, self.entities)}
        else:
            return self.entities


    def get_more_embeddings(self, new_sentences, new_labels=None, labels=True):
        """
        Doesn't store vectors, merely returns them.
        BUG - I NEED TO STORE AND QUERY THE PHRASES MODEL
        """
        doc_nouns = self.get_proper_nouns(new_sentences)

        if labels:
            return {x[0]:x[1] for x in zip(new_labels, self.get_entities(doc_nouns))}
        else:
            return self.get_entities(doc_nouns)
