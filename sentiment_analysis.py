'''
Perform Natural Language Processing (NLP) Sentiment Polarity Calculations of:
    DataFrame of clean comments -> analyze_comments()
    DataFrame of game threads with lists of clean comments -> analyze_threads()
'''
import nltk
import pandas as pd
import clean_text
import numpy as np
import re
from nltk.sentiment import vader

# Load SocialSent Dictionary (hockey.tsv)
polarities = pd.read_csv('./data/hockey.tsv', sep='\t', header=None, names=['word', 'polarity', 'conf'])
polarities_dict = dict(zip(polarities.word, polarities.polarity))


def analyze_comments(comment_data):
    '''
    Calculate comment sentiment in multiple ways and produce aggregrated sentiment polarity scores
    
    comment_data = DataFrame of which each row is a single cleaned comment
    '''
    comment_data['word_socialsent_polarity'] = comment_data.clean_comments.map(lambda x: get_word_socialsent_polarities(x))
    comment_data['vader_comment_polarity'] = comment_data.clean_comments.map(lambda x: get_vader_polarity(x))
    comment_data['full_socialsent_polarity'] = comment_data.clean_comments.map(lambda x: get_comment_polarities(x))
    comment_data['clean_context'] = comment_data.clean_comments.map(lambda x: get_game_context(x))
    comment_data['vader_context'] = comment_data.vader_comments.map(lambda x: get_game_context(x))
    comment_data['subreddit_context_word_socialset_polarity'] = comment_data.clean_context.map(lambda x: get_word_socialsent_polarities(x))
    comment_data['subreddit_context_polarity'] = comment_data.clean_context.map(lambda x: get_comment_polarities(x))
    comment_data['vader_context_polarity'] = comment_data.vader_context.map(lambda x: get_vader_polarity(x))
    comment_data['context_socialsent_pos_polarity'] = comment_data.clean_context.map(lambda x: get_comment_pos_polarities(x))
    comment_data['context_socialsent_no_ref_polarity'] = comment_data.clean_context.map(lambda x: get_context_no_ref_polarities(x))
    return comment_data


def get_word_socialsent_polarities(clean_comment):
    '''
    Find the setniment polarity for each word in a cleaned comment using Vader Scoring
    
    Output list of polarity scores for each word in comment
    '''
    if clean_comment == None:
        return np.nan
    tokenizer = nltk.tokenize.SpaceTokenizer()
    tokens = tokenizer.tokenize(clean_comment)
    word_polarities = []
    for token in tokens:
        try:
            pol = polarities_dict[token]
            word_polarities.append(pol)
        except KeyError:
            continue
    return word_polarities


def get_vader_polarity(vader_context):
    '''
    Calculate the sentiment polarity of a comment using VADER sentiment scoring
    
    Output = Sentiment Polarity
    '''
    if vader_context == None:
        return np.nan
    vader_sent = vader.SentimentIntensityAnalyzer()
    return vader_sent.polarity_scores(vader_context)['compound']


def get_comment_polarities(clean_comment):
    '''
    Calculate the sentiment polarity of a comment using SocialSent r/hockey sentiment scoring
    
    Output = Sentiment Polarity
    '''
    if clean_comment == None:
        return np.nan
    tokenizer = nltk.tokenize.SpaceTokenizer()
    tokens = tokenizer.tokenize(clean_comment)
    polarity = 0
    for token in tokens:
        try:
            pol = polarities_dict[token]
            polarity += pol
        except KeyError:
            continue
    return polarity


def get_game_context(clean_comment):
    '''
    Parse context of the word 'game' in comments
    
    Output = context of the word 'game'
    '''
    toker = nltk.tokenize.SpaceTokenizer()
    if clean_comment == None:
        return ''
    tokens = toker.tokenize(clean_comment)
    context_words = get_game_context_words(tokens)
    return context_words


def get_game_context_words(tokens, window_size=5):
    '''
    Parse context of the word 'game' in list of tokens
    
    tokens = Output of toker.tokenize()
    
    window_size = Number of workds before and after the word 'game' to include
    as context
    
    Output = string of words included in all game contexts of a given comment
    '''
    windows = []
    for index, token in enumerate(tokens):
        window = None
        if token == 'game':
            window = [index-window_size, index+window_size]
        if window != None:
            if window[0] < 0:
                window[0] = 0
            elif window[1] > len(tokens):
                window[1] = len(tokens)
            windows.append((window[0], window[1]))
    
    contexts = get_words(tokens, windows)
    contexts = ' '.join(contexts)
    return contexts


def get_words(tokens, windows):
    '''
    Parse context of the word 'game' in list of tokens
    
    tokens = Output of toker.tokenize()
    
    windows = Output of get_game_context_words()
    
    Output = List of different game contexts in a given comment
    '''
    contexts = [] 
    for window in windows:
        context = tokens[window[0]:window[1]]
        contexts.extend(context)
    return contexts


def get_comment_pos_polarities(clean_comment):
    '''
    Calculate the sentiment polarity of a comment using SocialSent r/hockey sentiment scoring
    while ignoring words with negative polarity scores
    
    Output = Sentiment Polarity
    '''
    if clean_comment == None:
        return np.nan
    tokenizer = nltk.tokenize.SpaceTokenizer()
    tokens = tokenizer.tokenize(clean_comment)
    polarity = 0
    for token in tokens:
        try:
            pol = polarities_dict[token]
            if pol > 0:
                polarity += pol
        except KeyError:
            continue
    return polarity

def get_context_no_ref_polarities(clean_comment):
    '''
    Calculate the sentiment polarity of a comment using SocialSent r/hockey sentiment scoring
    while ignoring comments with 'ref' in them
    
    Output = Sentiment Polarity
    '''
    if clean_comment == None:
        return np.nan
    if 'ref' in clean_comment:
        return 0
    tokenizer = nltk.tokenize.SpaceTokenizer()
    tokens = tokenizer.tokenize(clean_comment)
    polarity = 0
    for token in tokens:
        try:
            pol = polarities_dict[token]
            polarity += pol
        except KeyError:
            continue
    return polarity


def analyze_threads(total_data, drop_text=True):
    '''
    Calculate thread sentiment in multiple ways and produce aggregrated sentiment polarity scores
    
    total_data = DataFrame of which each row is a game thread
    '''
    total_data = total_data.copy()
    
    filter_fanbase(total_data)
    flair_diversity(total_data)
    
    total_data['full_vader_polarity'] = total_data['clean_comments'].map(lambda x: [get_vader_polarity(y) for y in x])
    total_data['vader_comment'] = total_data['full_vader_polarity'].map(lambda x: np.nansum(x))
    
    total_data['full_socialsent_polarity'] = total_data['clean_comments'].map(lambda x: [get_comment_polarities(y) for y in x])
    total_data['socialsent_comment'] = total_data['full_socialsent_polarity'].map(lambda x: np.nansum(x))

    total_data['game_context'] = total_data.clean_comments.map(lambda x: [get_game_context(y) for y in x])  
    
    total_data['context_vader_polarity'] = total_data['game_context'].map(lambda x: [get_vader_polarity(y) for y in x])
    total_data['vader_context'] = total_data['context_vader_polarity'].map(lambda x: np.nansum(x))

    total_data['context_socialsent_polarity'] = total_data['game_context'].map(lambda x: [get_comment_polarities(y) for y in x])
    total_data['socialsent_context'] = total_data['context_socialsent_polarity'].map(lambda x: np.nansum(x))
    
    total_data['pos_context_socialsent_polarity'] = total_data['game_context'].map(lambda x: [get_comment_pos_polarities(y) for y in x])
    total_data['positive_context'] = total_data['pos_context_socialsent_polarity'].map(lambda x: np.nansum(x))
    
    total_data['non_fan_game_context'] = total_data.non_fan_comments.map(lambda x: [get_game_context(y) for y in x])
    total_data['non_fan_context_socialsent_polarity'] = total_data['non_fan_game_context'].map(lambda x: [get_comment_polarities(y) for y in x])
    total_data['non-fan_context'] = total_data['non_fan_context_socialsent_polarity'].map(lambda x: np.nansum(x))

    total_data['no_ref_context_socialsent_polarity'] = total_data['game_context'].map(lambda x: [get_context_no_ref_polarities(y) for y in x])
    total_data['no_ref_context'] = total_data['no_ref_context_socialsent_polarity'].map(lambda x: np.nansum(x))

    total_data.drop(columns=['full_vader_polarity', 'full_socialsent_polarity',
                             'game_context', 'context_vader_polarity',
                             'context_socialsent_polarity', 'pos_context_socialsent_polarity',
                             'non_fan_comments', 'fan_comments', 'non_fan_game_context',
                             'non_fan_context_socialsent_polarity', 'no_ref_context_socialsent_polarity'],
                    inplace=True)
    
    if drop_text == True:
        total_data.drop(columns=['comments', 'clean_comments', 'author_flair'], inplace=True)
    return total_data


def filter_fanbase(total_data):
    '''
    Seperates comments based on the author flair. Comments posted by authors 
    whose flair is of a team playing in a given game are grouped as fan comments. 
    All other comments are grouped as non-fan comments
    
    total_data = DataFrame of which each row is a game thread
    '''
    non_fan_comments = []
    fan_comments = []
    for i in range(0, len(total_data)):
        coms = total_data.iloc[i]['clean_comments']
        flair = total_data.iloc[i]['author_flair']
        flair_coms = dict(zip(coms, flair))
        teams = [total_data.iloc[i]['home_team'], total_data.iloc[i]['away_team']]
        non_fan = []
        fan = []
        for com in coms:
            try:
                if flair_coms[com] in teams:
                    fan.append(com)
                else:
                    non_fan.append(com)
            except KeyError:
                    non_fan.append('None')
        non_fan_comments.append(non_fan)
        fan_comments.append(fan)
    total_data['non_fan_comments'] = non_fan_comments
    total_data['fan_comments'] = fan_comments
    return total_data


def flair_diversity(total_data):
    '''
    Calculates the percentage of comments on post-game threads that are from 
    non-fans. Outputs the Data with an additional column of this percentage ('non_fan%')
    '''
    flair_diversity = []
    for i in range(0, len(total_data)):
        post = total_data.iloc[i]
        try:
            fan_perc = len(post['fan_comments'])/len(post['comments'])
        except ZeroDivisionError:
            fan_perc = 0
        flair_diversity.append(fan_perc)
    total_data['fan%'] = flair_diversity
    return total_data


def get_sample_comment(comment_data, scoring=None, negative=False, perc=1, num_comments=1):
    '''
    Produces random comment samples from top positive/negative sentiment
    
    comment_data (DataFrame) = output of analyze_comments()
    scoring (string) = sentiment scoring metric (feature name)
    negative (Boolean, default = False) = If True, sample top negative comments, 
        if false, sample top positive comments
    perc (numeric, default = 1) = top percentage to sample from
    num_comments (integer, default = 1) = number of comments to produce
    '''
    top_number = int(((perc/100)*len(comment_data)))
    top_comments = comment_data.sort_values(by=scoring, ascending=negative).head(top_number)['clean_comments']
    print(np.random.choice(top_comments, size=num_comments))
