#!/usr/bin/env python
# coding: utf-8

# In[1]:
'''
To clean reddit comment text

For basic cleaning (punctuation, urls, bots, empty comments) use reduce_noise
    -Use for contextual approaches e.g. word2vec

For full lemmatization, use get_clean_comments
    -Use for context-blind approaches e.g. token-based sentiment analysis, TF-IDF
    
'''
import string
import logging
import re
import nltk
import numpy as np
from nltk.corpus import wordnet


def prep_for_vader(comments):
    comments = remove_urls(comments)
    # Remove bots
    comments = ['' if 'bot' in x else x for x in comments]
    # Replace empty comments with None
    comments = [None if x == '' else x for x in comments]
    return comments


# def lemma_comments(post_comments):
#     '''
#     Fully processes text. Removes noise (URLs, punctuation, bot comments, stop words)
#     and lemmatizes words
    
#     input = list of comment strings
    
#     output list of cleaned comment strings
#     '''
#     comments = reduce_noise(post_comments)
#     token_comments = tokenize(comments)
#     tagged_comments = tag_comments(token_comments)
#     no_stop_tagged_comments = remove_stop_words(tagged_comments)
#     lemma_comments = lemmatize(no_stop_tagged_comments)
#     final_comments = finalize_comments(lemma_comments)
#     return final_comments


def reduce_noise(comments):
    '''
    Removes puntuation, urls, bot comments, and empty comments
    
    input = list of comment strings
    
    output = list of cleaned comment strings 
    '''
    comments = clean_punctuation(comments)
    comments = remove_urls(comments)
    # Remove bots
    comments = ['' if 'bot' in x else x for x in comments]
    # Replace empty comments with None
    comments = [None if x == '' else x for x in comments]
    return comments


def clean_punctuation(comments):
    '''
    Remove punctuation and '\n' and '[deleted]' in comments
    
    comments = list of comment strings
    
    Output list of comments without punctuation
    '''
    clean_comments = []
    for comment in comments:
        comment = comment.replace('\n', ' ')
        comment = comment.replace('[deleted]','')
        comment = comment.translate(str.maketrans('', '', string.punctuation.replace("'","")))
        comment = comment.lower()
        clean_comments.append(comment)
    return clean_comments


def remove_urls(comments):
    '''
    Replace any comment with a URL with ''
    
    comments = list of comment strings
    
    Output list of comments without URLs
    '''
    no_urls = []
    for comment in comments:
        if len(re.findall(r"http\S+", comment)) == 0:
            no_urls.append(comment)
        else:
            no_urls.append('')
    return no_urls


def tokenize(comments):
    '''
    Tokenize comments by whitespace
    
    comments = list of comment strings
    
    Output list of lists of comment tokens
    '''
    token_comments = []
    for comment in comments:
        if type(comment) == str:
            tokens = nltk.tokenize.WhitespaceTokenizer().tokenize(comment)
            token_comments.append(tokens)
        else:
            token_comments.append(None)
    return token_comments


def tag_comments(token_comments):
    '''
    Tag comments with Part-of-Speach (POS) code
    
    token_comments = output of tokenize()
    
    Output list of lists of tagged comment tokens
    '''
    tagged_comments = []
    for comment in token_comments:
        if comment == None:
            tagged_comments.append(None)
        else:
            tags = nltk.pos_tag(comment)
            converted_tags = []
            for tag in tags:
                tag = list(tag)
                tag[1] = wordnet_pos_code(tag[1])
                tag = tuple(tag)
                converted_tags.append(tag)
            tagged_comments.append(converted_tags)
    return tagged_comments


def wordnet_pos_code(tag):
    '''
    Translates tags from nltk.pos_tag() to wordnet pos tags (For lemmatization)
    
    tag = tag output of nltk.pos_tag()
    
    output = wordnet pos tag
    '''
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''

    
def remove_stop_words(tagged_comments):
    '''
    Removes stop words
    
    tagged_comments = output of tag_comments()
    
    output = tagged comments with no stop words
    '''
    no_stop_tagged_comments = []
    for comment in tagged_comments:
        if comment == None:
            no_stop_tagged_comments.append(None)
        else:
            comment = [tag for tag in comment if not tag[1] == '']
            no_stop_tagged_comments.append(comment)
    return no_stop_tagged_comments


def lemmatize(no_stop_tagged_comments):
    '''
    Lemmatizes words 
    
    no_stop_tagged_comments = output of remove_stop_words()
    
    output = lemmatized comments
    '''
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemma_comments = []
    for comment in no_stop_tagged_comments:
        if comment == None:
            lemma_comments.append(None)
        else:
            lemmas = []
            for word in comment:
                lemma = lemmatizer.lemmatize(word[0], pos=word[1])
                lemmas.append(lemma)
            lemma_comments.append(lemmas)
    return lemma_comments


def finalize_comments(lemma_comments):
    '''
    Re-joins comment lemmas into a single string 
    
    lemma_comments = output of lemmatize()
    
    output = list of clean, lemmatized comments
    '''
    final_comments = []
    for lemma_comment in lemma_comments:
        if lemma_comment == None:
            final_comments.append(None)
        else:
            clean_comment = ' '.join(lemma_comment)
            final_comments.append(clean_comment)
    return final_comments
