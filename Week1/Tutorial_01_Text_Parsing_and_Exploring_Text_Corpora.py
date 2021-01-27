#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: Text Parsing and Exploring Text Corpora
# 
# In this notebook, we will cover:
#  - Tokenization
#  - Parts of Speech (PoS) tagging
#  - Parsing Techniques such as:
#   - Shallow Parsing
#   - Consituency Trees
#  - Basics of WordNet
#  - Application of these concepts on a given dataset/corpus 
# 
#  This notebook will use popular NLP packages such as:
#   - ``nltk``
#   - ``spacy``
# 
# We will also use popular data processing and handling packages such as ``numpy`` and ``pandas``
# 
# Let's get started!
# 
# __Note:__ We recommend running these notebooks in Google Colab so you don't have to install dependencies manually in your environment or face challenges with a few dependencies on windows environments.

# # Import Libraries and Setup Packages
import nltk
import spacy
import numpy as np
import pandas as pd
# download spacy models for English language
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
# 'en_core_web_sm'
# download PoS tagger for NLTK
nltk.download('averaged_perceptron_tagger')

sentence = "The brown fox is quick and he is jumping over the lazy dog"
sentence
# split/tokenize sentence into words
words = sentence.split()
# print the list of words tokenized from the input sentence
print(words)

pos_tags = nltk.pos_tag(sentence.split())

pos_tags

print(pos_tags)

# transform output into a pandas dataframe for ease of understanding
pd.DataFrame(pos_tags).T


# ## Spacy for POS Tagging
# 
# ``spacy`` also provides POS tagging capabilities out of the box.
# 
# - Step 1: Parse the sentence using the ``spacy``'s ``nlp`` object we created while importing spacy models
# - Step 2: Loop through each of the words which has a tag and corresponding location information attached

# In[ ]:


# get POS tags using spacy
spacy_pos_tagged = [(word, word.tag_, word.pos_) for word in nlp(sentence)]

# Format the output using pandas dataframe
pd.DataFrame(spacy_pos_tagged).T


# # Shallow Parsing
# 
# We extracted POS tags in the previous step. Shallow parsing is the next logical step which helps us in understanding the relationship betweek the POS tags.
# 
# More formaly, shallow parsing is a method to analyse a sentence to identify parts of speech and then link them to higher order units that have discrete grammatical meanings.
# 
# 
# 
# 

# In the hierarchy tree, groups of words make up phrases which form the third level in the syntax tree. By principle, phrases are assumed to have at least two or more words considering the pecking order of words ⟵phrases ⟵clauses ⟵symbols. 
# 
# There are five major categories of phrases which are described below.
# 
# - Noun Phrase (NP): These are phrases where a noun acts as the head word.
# 
# - Verb Phrase (VP): These phrases are lexical units which have a verb acting as the head word. 
# 
# - Adjective Phrase (ADJP): These are phrases whose head word is an adjective. Their main role is to describe or qualify nouns and pronouns in a sentence and they will be either placed before or after the noun or pronoun. 
# 
# - Adverb Phrase (ADVP): These phrases act like an adverb since the adverb acts as the head word in the phrase. Adverb phrases are used as modifiers for nouns, verbs or adverbs themselves by providing further details to describe or qualify them. 
#  
# - Prepositional Phrase (PP): These phrases usually contain a preposition as the head word and other lexical components like nouns, pronouns etc. It acts like an adjective or adverb describing other words or phrases. 
# 
# 
# These five major syntactic categories of phrases can be generated from words using several rules some of which we discussed above like utilizing syntax and grammars of different types.
# 

# In the following cell, we will :
# - Create a very general grammar rule consisting of Nouns, Adjectives, Prepositions and so on.
# - Use this rule to create a regex based sentence parser
# - We will then use the POS tags of our input sentence to build a shallow parse tree of the input
# 
# 
# We will be using ``nltk`` utilities for this exercise

# In[ ]:


grammar = '''
            NP: {<DT>?<JJ>?<NN.*>}  
            ADJP: {<JJ>}
            ADVP: {<RB.*>}
            PP: {<IN>}      
            VP: {<MD>?<VB.*>+}
          '''

pos_tagged_sent = nltk.pos_tag(sentence.split())
rp = nltk.RegexpParser(grammar)
shallow_parsed_sent = rp.parse(pos_tagged_sent)
print(shallow_parsed_sent)


# #### Visualize Parse Tree
# 
# The output from previous cell can be better understood using a visualization called as Parse Tree.
# 
# For folks using Colab to run this notebook, we would need to install a separate package called ``svgling`` to visualize the tree. For others, its optional

# In[ ]:


# this is used for handling tkinter issue on colab/jupyter
# svgling monkey-patches nltk tree draw
get_ipython().system('pip install svgling')

import svgling
# visualize shallow parse tree
svgling.draw_tree(shallow_parsed_sent)


# # Constituency Parsing
# Constituency parsing is the method which helps us understand a sentence as a combination of smaller phrases. A constituency parse tree breaks a text into smaller sub-phrases. Intermediate nodes in the tree are the phrases while the terminal nodes are the words in the sentence. Typically edges are left unlabeled in a constituency tree.
# 
# Phrase structure rules form the core of constituency grammars since they talk about syntax and rules which govern the hierarchy and ordering of the various constituents in the sentences. These rules cater to two things primarily. The generic representation of a phrase structure rule is S →A B which depicts that the structure S consists of constituents A and B and the ordering is A followed by B.
# 
# There are several phrase structure rules and we will explore them one by one to understand how exactly do we extract and order constituents in a sentence. The most important rule describes how to divide a sentence or a clause. The phrase structure rule denotes a binary division for a sentence or a clause as S → NP VP where S is the sentence or clause and it is divided into the subject, denoted by the Noun Phrase (NP) and the predicate, denoted by the Verb Phrase (VP). 
# 
# 
# 
# To build a constituency tree, we need to download the ``stanford parser`` and use it with the help of interfaces provided by the ``nltk`` library

# download the stanford parser jar file
get_ipython().system('wget https://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip')


# In[ ]:


# unzip the downloaded file to get the required parser jar files
get_ipython().system('unzip stanford-parser-full-2015-04-20.zip')


# In[ ]:


from nltk.parse.stanford import StanfordParser

# /Users/fritz/Python_Projects/Natural_Language_Processing/NLP_Statistics_Class_Jan2021/Week1/stanford-parser-full-2015-04-20/stanford-parser.jar
# /Users/fritz/Python_Projects/Natural_Language_Processing/NLP_Statistics_Class_Jan2021/Week1/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar
# Create a stanford parser object by setting location for parser and model jar files
scp = StanfordParser(path_to_jar='./Week1/stanford-parser-full-2015-04-20/stanford-parser.jar',
                   path_to_models_jar='./Week1/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar')

scp2 = nltk.parse.corenlp.CoreNLPParser()

# In[ ]:


# get the parse tree
result = list(scp.raw_parse(sentence))
print(result[0])


# ### Visualize Tree

# In[ ]:


# visualize constituency tree
svgling.draw_tree(result[0])


# From the above figure, you can conclude that our sentence has two main clauses or constituents which we had talked about earlier and they are joined together by a co-ordinating conjunction (and). 

# # Dependency Parsing
# 
# Dependency grammars always have a one-to-one relationship correspondence for each word in the sentence. There are two aspects to this grammar representation. One is the syntax or structure of the sentence and the other is the semantics obtained from the relationships denoted between the words. 
# 
# Each directed edge represents a specific type of meaningful relationship (also known as syntactic function) and we can annotate our sentence further showing the specific dependency relationship types between the words.
# 
# Instead of creating a tree with linear orders, you can also represent it with a normal graph since there is no concept of order of words in dependency grammar. We can leverage spacy to build us this dependency tree\graph for our sample sentence.
# 
# ``spacy`` provides an elegant and easy to use interface to generate similar output. It also provides a cleaner visualization using the ``displacy`` utility

# In[ ]:


from spacy import displacy

displacy.render(nlp(sentence), jupyter=True, 
                options={'distance': 100,
                         'arrow_stroke': 1.5,
                         'arrow_width': 8})


# # Corpus Analysis
# 
# We have covered a number of concepts so far. Let us now apply that understanding on a given corpus.
# 
# Text corpora is the plural form of 'text corpus' and can be defined as large and structured collection of texts or textual data. It usually consists of a body of written or spoken text, often stored in electronic form. This includes converting old historic text corpora from physical to electronic form so that it can be analyzed and processed with ease. The primary purpose of text corpora is to leverage them for linguistic as well as statistical analysis and to use them as data for building natural language processing tools. 
# 
# __Brown Corpus:__ This was the first million-word corpus for the English language, published by Kucera and Francis in 1961, also known as "A Standard Corpus of Present-Day American English". This corpus consists of text from a wide variety of sources and categories.
# 
# We will leverage the _brown_ corpus from ``nltk`` as our starting point

# We have already talked a bit about the Brown Corpus which was developed in 1961 at the Brown University. This corpus consists of texts from 500 sources and has been grouped into various categories. The following code snippet loads the Brown Corpus into the system memory and shows the various available categories.

# download the corpus
nltk.download('brown')
# load the Brown Corpus
from nltk.corpus import brown
# get total number of categories
print('Total Categories:', len(brown.categories()))

# ### Tokenize Sentences
# tokenized sentences
# this is already done/pre-processed by nltk
brown.sents(categories='mystery')

# ### POS Tagging
# this is already done/pre-processed by nltk
brown.tagged_sents(categories='mystery')

# ## Create Sentences
# 
# Since the brown corpus already contains tokenized sentences, let us try and create their original structure and print the same
# get sentences in natural form
sentences = brown.sents(categories='mystery')

sentences_joined = [' '.join(sentence_token) for sentence_token in sentences]

sentences3 = []
for i in range(len(sentences2))
for i, sent in enumerate(sentences2):
    try:
        sentences3.append(sent)
    except AssertionError:
        continue



sentences[0:5] # viewing the first 5 sentences


# ### Get List of Nouns
# To get a list of nouns from the preprocessed corpus, all we need to do is look for POS tags which signify Nouns 
# 
# (_hint: NP and NN_)

# In[ ]:


# get tagged words
tagged_words = brown.tagged_words(categories='mystery')

# get nouns from tagged words
nouns = [(word, tag) for word, tag in tagged_words if any(noun_tag in tag for noun_tag in ['NP', 'NN'])]

nouns[0:10] # view the first 10 nouns


# ## Analyze Nouns
# 
# - There are quite a lot of nouns in this dataset
# - Let us understand which are the most frequently occuring ones

# In[ ]:


# build frequency distribution for nouns
nouns_freq = nltk.FreqDist([word for word, tag in nouns])

# view top 10 occuring nouns
nouns_freq.most_common(10)


# ## Additional Corpora to Analyze

# In[ ]:


nltk.download('reuters')
nltk.download('punkt')


# In[ ]:


# load the Reuters Corpus
from nltk.corpus import reuters

# total categories
print('Total Categories:', len(reuters.categories()))


# In[ ]:


# get sentences in housing and income categories
sentences = reuters.sents(categories=['housing', 'income'])
sentences = [' '.join(sentence_tokens) for sentence_tokens in sentences]
sentences[0:5]  # view the first 5 sentences


# In[ ]:


# file-id based access
print(reuters.fileids(categories=['housing', 'income']))


# In[ ]:


print(reuters.sents(fileids=[u'test/16118', u'test/18534']))


# # WordNet
# 
# WordNet is a lexical database of semantic relations between words in more than 200 languages. WordNet links words into semantic relations including synonyms, hyponyms, and meronyms. The synonyms are grouped into synsets with short definitions and usage examples
# 
# WordNet database is also available through ``nltk`` library. It provides a straightforward and easy to use interface to utilize various features of the wordnet database.

# In[ ]:


# download wordnet
nltk.download('wordnet')


# In[ ]:


# load the Wordnet Corpus
from nltk.corpus import wordnet as wn


# ### Get Synset for a sample word
# 
# **Synsets**: group of synonyms are termed as synsets.

# In[ ]:


word = 'hike' # taking hike as our word of interest
# get word synsets
word_synsets = wn.synsets(word)
word_synsets


# ## Synset Details
# 
# Wordnet also contains definitions, most used POS tag and a few examples to showcase usage of a given word's synonyms.

# In[ ]:


# get details for each synonym in synset
for synset in word_synsets:
    print(('Synset Name: {name}\n'
           'POS Tag: {tag}\n'
           'Definition: {defn}\n'
           'Examples: {ex}\n').format(name=synset.name(),
                                      tag=synset.pos(),
                                      defn=synset.definition(),
                                      ex=synset.examples()))
          

