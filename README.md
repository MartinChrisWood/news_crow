# Exploring Different Document Clustering Methods

## Testing network-based methods vs word embeddings vs LDA


### 01_topic_pipeline_network

- Extract all Nouns (names) from all articles using Part-Of-Speech tagging (SpaCy)
- Apply NPMI to identify nouns that should be conjoined to phrases and treated as a single token
- Create an adjacency matrix based on co-occurence of nouns
- Generate a graph from the adjacency matrix, documents become nodes, and non-directional edges exist where > 2 nouns in common are found

For clustering, testing Label Propagation, finding maximally connected cliques, BigClam, and the Louvain community detection algorithm.  A complication with cliques and BigClam is that cliques/BigClam sets are not mutually exclusive - a document can be a member of multiple overlapping cliques.  This is natural for data like this, since many stories will be exploring the interactions of different entities (eg; the "prime minister" and "immigration") and genuinely reflects the corpus' subject matters.  Not sure how to evaluate overlapping clusters yet.


### 02_topic_pipeline_HDBSCAN

Pre-processing will include dropping duplicate stories, removing stopwords and lemmatizing the remainder. Additionally, multi-word phrases will be detected and conjoined using Normalised Pointwise Mutual Information (NPMI).  A Continuous-Bag-Of-Words (CBOW) Word2Vec model will be trained on the corpus.  Minimum word count allowed will be set to 1, so that every word found in the training data will be found in the model vocabulary. Vectors are length 100.  Document representations will be created by element-wise averaging of their tokens' vectors.

To create clusters to be treated as topics, the HDBSCAN algorithm will be applied to the document vectors.  Two approaches to applying it will be tried; clustering the vectors directly, and clustering the vectors after applying dimensionality reduction using PCA.

The overall pipeline here has a lot of hyperparameters for both HDBSCAN and Word2Vec.


### 03_topic_pipeline_LDA

LDA functions using a Bag-Of-Words (BOW) or Term Frequency - Inverse Document Frequency (TF-IDF) representation.  Pre-processing will include dropping duplicate stories, removing stopwords and lemmatizing the remainder.  Additionally, multi-word phrases will be detected and conjoined using Normalised Pointwise Mutual Information (NPMI).  NPMI functions by calculating how probable it is that two words are found adjacent in a document versus how probable that they are found apart in a document.  The effect is to condense the corpus, joining likely words into their more meaningful phrases and treating those phrases as a single token.  This is more potent than simply fitting an LDA model to both words and all n-grams, as it can act to disambiguate ("domestic\_abuse" and "domestic\_violence" instead of "domestic", "abuse" and "violence") and reduce noise in the model.

LDA requires that the number of topics desired be specified.  An exhaustive search of the number of topics will be performed, with each model being evaluated on perplexity and coherence to find an optimal solution.


### 04_clustering_metrics

The clustering/topic modelling methods will be compared on multiple different measures of topic coherence.  The decision may not be clear-cut;  both the network methods and HDBSCAN allow for unassigned outliers which may make that model appear more coherent by expediently dropping any stories that do not fit well.  LDA on the other hand is used to cluster by assigning each document to their most probable topic, and a document's topic probabilities are constrained to sum to 1, so LDA does not designate outliers.  Methods of extractive summarization will later be used to check the quality of clusters, alongside quantitative coherence measures and basic statistics on the topic size/coherence distributions.


### 05_text_summarisation

An example of using the TextRank algorithm to summarize text using pre-trained vector models (GLoVe, Doc2Vec) to find and extract the most representative sentences from a short text.

Using pre-trained models because if it works with those such a method could be applied to very small corpuses without issue.

### 06_text_summarisation_bu_analogy

Variant of the above; rather than using text rank, manually define some generic sentence fragments designed to reflect key/summary sentences (eg; "in summary", "to conclude"), and use sentence similarity to these to find the most useful overview/summary/conclusion sentences.

This hasn't worked well yet.


