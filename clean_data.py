'''
Convert and parse raw json outputs of scrape_reddit_posts.total_scrape() and 
links games with stats from www.hockey-reference.com

Use get_clean_data() to clean data
'''

from clean_text import reduce_noise, prep_for_vader
from nltk.corpus import stopwords
from textblob import TextBlob
import logging
import nltk
import numpy as np
import pandas as pd
import re


def get_clean_data(raw_reddit_data, start_date, end_date, hockey_ref_stats=True):
    '''
    Parse output of scrape_reddit_posts.total_scrape() to retrieve game information, 
    comments, and author flair and cleans data for analysis
    
    raw_reddit_data = output of scrape_reddit_posts.total_scrape()
    
    Date Format = 'YEAR-MONTH-DAY'; e.g. '2017-10-15' 
    
    hockey_ref_stats = Boolean (Default = True). True uses statistics from www.hockey-reference.com (determined to be more reliable), False uses statistics parsed from text of reddit post-game post
    '''
    # Extract relevant data
    extracted_post_data = extract_data(raw_reddit_data)
    
    # Convert to dataframe, clean, add features (date, polarity)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    reddit_posts = clean_post_comments(extracted_post_data, start_date, end_date)

    # Extract game statistics and clean
    reddit_posts, game_stats = remove_sparse_stats(reddit_posts)
    game_stats = clean_game_stats(game_stats)
    
    # Merge data
    clean_reddit_data = reddit_posts.merge(game_stats,
                                           left_index=True,
                                           right_index=True)
    clean_reddit_data = add_features(clean_reddit_data)
    clean_reddit_data.drop(columns=['away_1st_goals', 'away_2nd_goals',
                                     'away_3rd_goals', 'home_1st_goals',
                                     'home_2nd_goals', 'home_3rd_goals'], inplace=True)
    
    # Clean text of comments and retrieve author flair
    clean_reddit_data['clean_comments'] = clean_reddit_data.comments.map(
        lambda x: reduce_noise(x))
    clean_reddit_data['vader_comments'] = clean_reddit_data.comments.map(
        lambda x: prep_for_vader(x))
    clean_reddit_data['author_flair'] = clean_reddit_data['author_flair'].map(
        lambda x: get_flairs(x))
    
    # Replace statistics parsed by remove_bogus_stats() with statistics scraped from 
    # www.hockey-reference.com
    if hockey_ref_stats == True:
        clean_reddit_data = append_hockey_ref_stats(clean_reddit_data)

    # Correct 3-Letter Abbreviation of Vegas Golden Knights from 'VEG' to 'VGK'
    clean_reddit_data['home_team'] = ['VGK' if x == 'VEG' else x for x
                                       in clean_reddit_data['home_team']]
    clean_reddit_data['away_team'] = ['VGK' if x == 'VEG' else x for x
                                       in clean_reddit_data['away_team']]
    
    # Drop Duplicates
    clean_reddit_data.drop_duplicates(subset=['date', 'home_team', 'away_team'], inplace=True)
    clean_reddit_data.reset_index(drop=True, inplace=True)
    return clean_reddit_data


def extract_data(reddit_posts_raw):
    '''
    Parses json files from output of scrape_reddit_posts.total_scrape() to retrieve game, 
    comment, and comment author information
    
    reddit_posts_raw = output of scrape_reddit_posts.total_scrape()
    '''
    extracted_post_data = []
    for i in range(0, len(reddit_posts_raw)):
        post = reddit_posts_raw.iloc[i]
        title = post[0]['data']['children'][0]['data']['title']
        post_comments = []
        author_flairs = []
        for i in post[1]:
            try:
                post_comments.extend(get_all_comments(i['data']['children']))
                author_flairs.extend(get_all_flair(i['data']['children']))
            except:
                post_comments.append(i['data']['body'])
        stats = get_stats(post[0])
        extracted_post_data.append(
            (title, stats, post_comments, author_flairs))
    return extracted_post_data


def get_all_comments(post_comments_json):
    '''
    Output all comments from a reddit post

    post_comments_json = response.json()[1]['data']['children'] where response is reddit API response
    '''
    comments = []
    # Resursively search through json to find all comments in thread, including reply chains
    for com in post_comments_json:
        if com['kind'] == 'more':
            continue
        comments.append(com['data']['body'])
        if type(com['data']['replies']) == str:
            continue
        else:
            comments.extend(get_all_comments(
                com['data']['replies']['data']['children']))
    return comments


def get_all_flair(post_comments_json):
    '''
    Output all flair from a reddit post

    post_comments_json = response.json()[1]['data']['children'] where response is reddit API response
    '''
    flair = []
    # Resursively search through json to find all comments in thread, including reply chains
    for com in post_comments_json:
        if com['kind'] == 'more':
            continue
        flair.append(com['data']['author_flair_text'])
        if type(com['data']['replies']) == str:
            continue
        else:
            flair.extend(get_all_flair(
                com['data']['replies']['data']['children']))
    return flair


def get_stats(post):
    '''
    Parse all game statistics from post text
    
    post_comments_json = response.json()[1]['data']['children'] where response is reddit API response
    '''    
    post_text = post['data']['children'][0]['data']['selftext']
    try:
        title_goals = post_text.split(sep='\n')[2]
        home_goals = post_text.split(sep='\n')[4]
        away_goals = post_text.split(sep='\n')[5]
        title_stats = post_text.split(sep='\n')[7]
        home_info = post_text.split(sep='\n')[9]
        away_info = post_text.split(sep='\n')[10]

        pattern = '(.*?)\|'
        home_stats = re.findall(pattern, home_info)[1:]
        away_stats = re.findall(pattern, away_info)[1:]
        home_goals = re.findall(pattern, home_goals)[1:]
        away_goals = re.findall(pattern, away_goals)[1:]
        title_stats_names = re.findall(pattern, title_stats)[1:]
        title_goals_names = re.findall(pattern, title_goals)[1:]
        
        num_fights = post_text.count('Fight')/2
        
        columns = ['away_' + name.lower().replace(' ', '_') for name in title_stats_names] + \
        ['home_' + name.lower().replace(' ', '_') for name in title_stats_names] + \
        ['away_' + name.lower().replace(' ', '_') + '_goals' for name in title_goals_names] + \
        ['home_' + name.lower().replace(' ', '_') + '_goals' for name in title_goals_names] + \
        ['num_fights']
        return dict(zip((columns), (home_stats + 
                                    away_stats + 
                                    home_goals + 
                                    away_goals +
                                    [num_fights])))
    except IndexError:
        return np.nan


def clean_post_comments(extracted_post_data, start_date, end_date):
    '''
    Cleans posts to only include bot-created Post Game Threads within the desired
    time period
    
    extracted_post_data = Output of extract_data()
    
    Date Format = 'YEAR-MONTH-DAY'; e.g. '2017-10-15'
    '''
    reddit_posts = pd.DataFrame(extracted_post_data)
    reddit_posts.columns = ['game', 'stats', 'comments', 'author_flair']
    # Make sure all posts are bot-posted
    reddit_posts = reddit_posts[reddit_posts['game'].map(lambda x: 'Post Game Thread:' in x)]
    # Retrieve/Clean Date
    reddit_posts['date'] = reddit_posts['game'].map(lambda x: x[-len('xx xxx xxxx'):])
    reddit_posts['date'] = reddit_posts['date'].map(lambda x: convert_to_datetime(x))
    reddit_posts.dropna(axis=0, subset=['date'], inplace=True)
    # Subset
    reddit_posts = reddit_posts.loc[(reddit_posts['date'] >= start_date) &
                                    (reddit_posts['date'] <= end_date)]
    reddit_posts.dropna(inplace=True)
    return reddit_posts


def convert_to_datetime(date_string):
    '''
    Converts string of date from reddit post to datetime type, returns NaN
    if error raised
    
    date_string = string of parsed date with format=('%d %b %Y')
    '''
    try:
        date_string = pd.to_datetime(date_string, format='%d %b %Y')
        return date_string
    except:
        return np.nan


def remove_sparse_stats(reddit_posts):
    '''
    Removes columns and rows of data that contain high degree of missing data,
    a result of poorly parsed statistics. 
    
    reddit_posts = output of clean_post_comments()
    '''
    sparse_columns = []
    sparse_rows = []

    reddit_posts.reset_index(drop=True, inplace=True)

    game_stats = pd.DataFrame(list(reddit_posts['stats']))
    
    refined_columns_list = list(game_stats.columns)
    
    # Protect sparse OT/SO statistics from removal for use in add_features()
    refined_columns_list.remove('away_ot_goals')
    refined_columns_list.remove('away_so_goals')
    refined_columns_list.remove('home_ot_goals')
    refined_columns_list.remove('home_so_goals')

    # Remove columns with many missing values
    for column in refined_columns_list:
        if sum(game_stats[column].isna()) > 1000:
            sparse_rows.extend(
                game_stats.index[game_stats[column].isna() == False])
            sparse_columns.append(column)

    game_stats.drop(index=list(set(sparse_rows)), inplace=True)
    game_stats.drop(columns=sparse_columns, inplace=True)

    reddit_posts.drop(index=list(set(sparse_rows)), inplace=True)
    reddit_posts.drop(columns=['stats', 'game'], inplace=True)
    return reddit_posts, game_stats


def clean_game_stats(game_stats):
    '''
    Cleans and parses game statistics and info parsed from reddit posts
    
    game_stats = second output of remove_sparse_stats() (i.e. remove_sparse_stats()[1])
    '''
    game_stats['away_team'] = game_stats['away_team'].apply(lambda x: re.findall(pattern='/r/(.*?)\)+', string=x)[0] if type(x) is str else np.nan)
    game_stats.dropna(subset=['away_team'], inplace=True)
    game_stats['home_team'] = game_stats['home_team'].apply(
        lambda x: re.findall(pattern='/r/(.*?)\)+', string=x)[0])
    subreddits = ['bluejackets', 'tampabaylightning', 'habs', 'floridapanthers', 'predators',
                  'sanjosesharks', 'losangeleskings', 'bostonbruins', 'coloradoavalanche',
                  'ottawasenators', 'dallasstars', 'calgaryflames', 'wildhockey',
                  'newyorkislanders', 'rangers', 'sabres', 'caps', 'penguins', 'winnipegjets',
                  'canucks', 'flyers', 'hawks', 'canes', 'anaheimducks', 'edmontonoilers',
                  'stlouisblues', 'detroitredwings', 'leafs', 'goldenknights', 'devils',
                  'coyotes']
    abbreviations = ['CBJ', 'TBL', 'MTL', 'FLA', 'NSH', 'SJS', 'LAK', 'BOS', 
                     'COL', 'OTT', 'DAL', 'CGY', 'MIN', 'NYI', 'NYR', 'BUF', 
                     'WSH', 'PIT', 'WPG', 'VAN', 'PHI', 'CHI', 'CAR', 'ANA',
                     'EDM', 'STL', 'DET', 'TOR', 'VEG', 'NJD', 'ARI']
    name_to_code = dict(zip(subreddits, abbreviations))
    game_stats['away_team'] = game_stats['away_team'].apply(
        lambda x: name_to_code[x])
    game_stats['home_team'] = game_stats['home_team'].apply(
        lambda x: name_to_code[x])
    game_stats['away_pp_goals'] = game_stats['away_power_plays'].map(
        lambda x: re.findall(pattern='(.*?)/', string=x)[0])
    game_stats['away_pp'] = game_stats['away_power_plays'].map(
        lambda x: re.findall(pattern='\/(.*)', string=x)[0])
    game_stats['home_pp_goals'] = game_stats['home_power_plays'].map(
        lambda x: re.findall(pattern='(.*?)/', string=x)[0])
    game_stats['home_pp'] = game_stats['home_power_plays'].map(
        lambda x: re.findall(pattern='\/(.*)', string=x)[0])
    game_stats['away_fo_wins'] = game_stats['away_fo_wins'].map(
        lambda x: re.findall(pattern='(.*)%', string=x)[0])
    game_stats['home_fo_wins'] = game_stats['home_fo_wins'].map(
        lambda x: re.findall(pattern='(.*)%', string=x)[0])
    game_stats.drop(columns=['away_power_plays', 'home_power_plays',
                             'away_teams_goals', 'home_teams_goals'], inplace=True)
    
    numeric_columns = ['away_1st_goals', 'away_2nd_goals', 'away_3rd_goals',
                       'away_blocked', 'away_fo_wins', 'away_giveaways',
                       'away_hits', 'away_shots',
                       'away_takeaways', 'away_total_goals', 'num_fights',
                       'home_1st_goals', 'home_2nd_goals', 'home_3rd_goals',
                       'home_blocked', 'home_fo_wins', 'home_giveaways',
                       'home_hits', 'home_shots',
                       'home_takeaways', 'home_total_goals',
                       'away_pp_goals', 'away_pp', 'home_pp_goals', 'home_pp']

    game_stats[numeric_columns] = game_stats[numeric_columns].applymap(lambda x: pd.to_numeric(x))
    return game_stats


def add_features(clean_reddit_data):
    '''
    Adds engineered game statistics (e.g. differentials, game sums, and
    boolean ot/so/combacks columns)
    
    clean_reddit_data = merged Dataframe of 1st output of remove_sparse_stats() 
    (i.e. remove_sparse_stats()[0]) and output of clean_game_stats()
    '''
    clean_reddit_data['num_comments'] = clean_reddit_data['comments'].map(lambda x: len(x))
    clean_reddit_data['total_goals'] = (clean_reddit_data['home_total_goals'] + 
                                        clean_reddit_data['away_total_goals'])
    clean_reddit_data['total_shots'] = (clean_reddit_data['home_shots'] + 
                                        clean_reddit_data['away_shots'])
    clean_reddit_data['total_hits'] = (clean_reddit_data['home_hits'] +
                                       clean_reddit_data['away_hits'])
    try:
        clean_reddit_data['total_pp'] = (clean_reddit_data['home_pp'] +
                                         clean_reddit_data['away_pp'])
    except KeyError:
        clean_reddit_data['total_pp'] = (clean_reddit_data['home_power_play'] + 
                                         clean_reddit_data['away_power_play'])
            
    clean_reddit_data['total_pp_goals'] = (clean_reddit_data['home_pp_goals'] +
                                           clean_reddit_data['away_pp_goals'])
    clean_reddit_data['goal_diff'] = np.abs(clean_reddit_data['home_total_goals'] -
                                            clean_reddit_data['away_total_goals'])

    clean_reddit_data['period_lead_changes'] = clean_reddit_data.apply(lambda x:                                                                         calculate_num_period_lead_changes(x), axis=1)
                                         
    clean_reddit_data['comeback?'] = clean_reddit_data['period_lead_changes'].map(lambda x: 
                                                                                  1 if x % 2
                                                                                  else 0)
    clean_reddit_data['ot?'] = clean_reddit_data['away_ot_goals'].map(lambda x: 
                                                                      1 if x in [0, 1] 
                                                                      else 0)
    clean_reddit_data['so?'] = clean_reddit_data['away_so_goals'].map(lambda x: 
                                                                      1 if x in [0, 1] 
                                                                      else 0)
    return clean_reddit_data


def calculate_num_period_lead_changes(post):
    '''
    Calculates number of lead changes in a game
    
    post =  a row in clean_reddit_data DataFrame
    '''
    count = 0
    away_goals = post['away_1st_goals']
    home_goals = post['home_1st_goals']
    leader1 = find_leader(post, away_goals, home_goals)
    if leader1 != None:
        count += 1
    away_goals += post['away_2nd_goals']
    home_goals += post['home_2nd_goals']
    leader2 = find_leader(post, away_goals, home_goals)
    if leader2 != leader1:
        count += 1
    away_goals += post['away_3rd_goals']
    home_goals += post['home_3rd_goals']
    leader3 = find_leader(post, away_goals, home_goals)
    if leader3 != leader1:
        count += 1
    return count


def find_leader(post, away_goals, home_goals):
    '''
    Finds the leader of a game at a certain point
    '''
    leader = None
    if away_goals > home_goals:
        leader = post['away_team']
    elif away_goals < home_goals:
        leader = post['home_team']
    else:
        pass
    return leader


def get_flairs(sample_flair):
    '''
    Parses flair from list of raw flair
    
    sample_flair = list of raw flair
    '''
    flairs = []
    for i in sample_flair:
        pattern = ':(.*?)-+'
        try:
            flair = re.findall(pattern, i)[1].strip()
            flairs.append(flair)
        except (TypeError, IndexError):
            flairs.append('None')
    return flairs

def append_hockey_ref_stats(clean_reddit_data):
    '''
    Replace statistics from reddit posts with statistics from www.hockey-reference.com
    '''
    
    clean_reddit_data.drop(columns=['away_blocked', 'away_fo_wins', 'away_giveaways', 
                                         'away_hits', 'away_ot_goals', 'away_shots',
                                         'away_so_goals', 'away_takeaways', 'away_total_goals',
                                         'home_blocked', 'home_fo_wins', 'home_giveaways',
                                         'home_hits', 'home_ot_goals', 'home_shots',
                                         'home_so_goals', 'home_takeaways', 'home_total_goals',
                                         'away_pp_goals', 'away_pp', 'home_pp_goals', 
                                         'home_pp'], inplace=True)
    
    hockey_ref_game_data = clean_hockey_ref_game_data()
    hockey_ref_team_data = clean_hockey_ref_team_data()

    hockey_ref_data = pd.merge(hockey_ref_game_data, hockey_ref_team_data, on=[
                               'home_team', 'away_team', 'date'], how='inner')
    total_data = pd.merge(clean_reddit_data.drop(columns=['total_goals', 'total_shots', 'total_hits', 'total_pp', 'total_pp_goals', 'goal_diff', 'period_lead_changes', 'ot?', 'so?']),
                          hockey_ref_data,
                          left_on=['date', 'away_team', 'home_team'],
                          right_on=['date', 'away_team', 'home_team'])
    return total_data

def clean_hockey_ref_game_data():
    '''
    Collect hockey-reference game data (outputs of scrape_hockey_reference.scrape_game_data())
    and clean 
    
    Output DataFrame of hockey-reference game data
    '''
    hockey_ref_game_data_2018 = pd.read_pickle('./data/all_game_data_2018')
    hockey_ref_game_data_2019 = pd.read_pickle('./data/all_game_data_2019')

    hockey_ref_game_data = pd.concat(
        [hockey_ref_game_data_2018, hockey_ref_game_data_2019], ignore_index=True)

    hockey_ref_game_data.rename(
        columns={'visitor': 'away_team', 'home': 'home_team'}, inplace=True)

    hockey_ref_game_data.attendance = hockey_ref_game_data.attendance.map(
        lambda x: x.replace(',', ''))
    hockey_ref_game_data.attendance = pd.to_numeric(
        hockey_ref_game_data.attendance)
    hockey_ref_game_data.len_game = hockey_ref_game_data.len_game.map(
        lambda x: pd.to_datetime(x).hour*60 + pd.to_datetime(x).minute)
    hockey_ref_game_data.date = pd.to_datetime(hockey_ref_game_data.date)
    hockey_ref_game_data.drop(
        columns=['visitor_goals', 'home_goals', 'ot/so'], inplace=True)
    return hockey_ref_game_data


def clean_hockey_ref_team_data():
    '''
    Collect individual teams' hockey-reference game data (outputs of
    scrape_hockey_reference.scrape_team_data()) and clean and generate 
    differential and overall game statistiscs
    
    Output DataFrame of hockey-reference team-specific data
    '''
    hockey_ref_team_data_2018 = pd.read_pickle('./data/team_game_data_2018')
    hockey_ref_team_data_2019 = pd.read_pickle('./data/team_game_data_2019')

    hockey_ref_team_data = hockey_ref_team_data_2018 + hockey_ref_team_data_2019

    dfs = []

    for i in hockey_ref_team_data:
        i[1]['team'] = i[0]
        dfs.append(i[1])

    hockey_ref_team_data = pd.concat(dfs)

    hockey_ref_team_data['win/loss'] = hockey_ref_team_data['win/loss'].map(
        lambda x: 1 if x == 'W' else 0)

    hockey_ref_team_data['ot/so'] = hockey_ref_team_data['ot/so'].map(
        lambda x: 2 if x == 'SO' else 1 if x == 'OT' else 0)

    hockey_ref_team_data.drop(columns=['games_played'], inplace=True)
    hockey_ref_team_data.rename(
        columns={'team': 'home_team', 'opponent': 'away_team'}, inplace=True)
    hockey_ref_team_data.rename(columns={'goals_for': 'home_goals',
                                         'goals_against': 'away_goals',
                                         'team_sog': 'home_sog',
                                         'team_pim': 'home_pim',
                                         'team_pp_goal': 'home_pp_goal',
                                         'team_pp_oppurtunities': 'home_pp_opportunities',
                                         'team_shorthanded_goals': 'home_shorthanded_goals',
                                         'opponent_sog': 'away_sog',
                                         'opponent_pim': 'away_pim',
                                         'opponent_pp_goal': 'away_pp_goal',
                                         'opponent_pp_oppurtunities': 'away_pp_opportunities',
                                         'opponent_shorthanded_goals': 'away_shorthanded_goals'},
                                inplace=True)

    for i in ['home_goals', 'away_goals',
              'win/loss', 'ot/so', 'home_sog', 'home_pim', 'home_pp_goal',
              'home_pp_opportunities', 'home_shorthanded_goals', 'away_sog',
              'away_pim', 'away_pp_goal', 'away_pp_opportunities',
              'away_shorthanded_goals', 'corsi_for', 'corsi_against', 'corsi_for%',
              'fenwick_for', 'fenwick_against', 'fenwick_for%', 'faceoff_wins',
              'faceoff_loses', 'faceoff%', 'off_zone_start%', 'pdo']:
        hockey_ref_team_data[i] = pd.to_numeric(hockey_ref_team_data[i])
    
    # Sum team-specific statistics
    hockey_ref_team_data['total_faceoffs'] = (hockey_ref_team_data['faceoff_wins'] +
                                              hockey_ref_team_data['faceoff_loses'])
    hockey_ref_team_data['total_sog'] = hockey_ref_team_data.home_sog + \
        hockey_ref_team_data.away_sog
    hockey_ref_team_data['total_pp_goals'] = hockey_ref_team_data.home_pp_goal + \
        hockey_ref_team_data.away_pp_goal
    hockey_ref_team_data['total_goals'] = hockey_ref_team_data.home_goals + \
        hockey_ref_team_data.away_goals
    hockey_ref_team_data['total_pp_opportunity'] = hockey_ref_team_data.home_pp_opportunities + \
        hockey_ref_team_data.away_pp_opportunities
    hockey_ref_team_data['total_sh_goals'] = hockey_ref_team_data.home_shorthanded_goals + \
        hockey_ref_team_data.away_shorthanded_goals
    
    # Subtract team-specific statistics differential
    hockey_ref_team_data['sog_diff'] = np.abs(hockey_ref_team_data.home_sog
                                              - hockey_ref_team_data.away_sog)
    hockey_ref_team_data['pp_goals_diff'] = np.abs(hockey_ref_team_data.home_pp_goal
                                                   - hockey_ref_team_data.away_pp_goal)
    hockey_ref_team_data['goal_diff'] = np.abs(hockey_ref_team_data.home_goals
                                               - hockey_ref_team_data.away_goals)
    hockey_ref_team_data['pp_opportunity_diff'] = np.abs(hockey_ref_team_data.home_pp_opportunities -
                                                         hockey_ref_team_data.away_pp_opportunities)
    hockey_ref_team_data['sh_goals_diff'] = np.abs(hockey_ref_team_data.home_shorthanded_goals -
                                                   hockey_ref_team_data.away_shorthanded_goals)
    
    # Calculate normalized team-specific differential 
    # Non-normalized differential was chosen over normalized 
#     hockey_ref_team_data['sog_norm_diff'] = (hockey_ref_team_data['sog_diff']/
#                                              hockey_ref_team_data['total_sog'])                                        
#     hockey_ref_team_data['pp_goals_norm_diff'] = (hockey_ref_team_data['pp_goals_diff']/
#                                                   hockey_ref_team_data['total_pp_goals'])    
#     hockey_ref_team_data['goal_norm_diff'] = (hockey_ref_team_data['goal_diff']/
#                                               hockey_ref_team_data['total_goals'])    
#     hockey_ref_team_data['pp_opportunity_norm_diff'] = (hockey_ref_team_data['pp_opportunity_diff']/
#                                                         hockey_ref_team_data['total_pp_opportunity'])    
#     hockey_ref_team_data['sh_goals_norm_diff'] = (hockey_ref_team_data['sh_goals_diff']/
#                                                   hockey_ref_team_data['total_sh_goals'])
#     hockey_ref_team_data.drop(columns=['sog_diff', 'pp_goals_diff', 'goal_diff', 'pp_opportunity_diff', 'sh_goals_diff'],
#                               inplace=True)
    
    # Normalize Percentages (Only care about deviation from 50%)
    hockey_ref_team_data['corsi_for%'] = hockey_ref_team_data['corsi_for%'].map(lambda x: (100-x) if x < 50 else x)
    hockey_ref_team_data['fenwick_for%'] = hockey_ref_team_data['fenwick_for%'].map(lambda x: (100-x) if x < 50 else x)
    hockey_ref_team_data['faceoff%'] = hockey_ref_team_data['faceoff%'].map(lambda x: (100-x) if x < 50 else x)
    hockey_ref_team_data['off_zone_start%'] = hockey_ref_team_data['off_zone_start%'].map(lambda x: (100-x) if x < 50 else x)

    hockey_ref_team_data.rename(columns={'goals_for': 'home_goals',
                                         'goals_against': 'away_goals',
                                         'corsi_for': 'home_corsi',
                                         'corsi_against': 'away_corsi',
                                         'fenwick_for': 'home_fenwick',
                                         'fenwick_against': 'away_fenwick',
                                         'faceoff_wins': 'home_faceoff_wins',
                                         'faceoff_loses': 'away_faceoff_wins'}, inplace=True)

    hockey_ref_team_data.replace(to_replace=np.nan, value=0, inplace=True)
    return hockey_ref_team_data


def extract_comment_data(data):
    '''
    Generates DataFrame in which each row is an individual comment
    
    data = Output of get_clean_data()
    '''
    all_comments = []
    vader_comments = []
    fans = []
    flair = []
    fan = []

    for i in range(0, len(data)):
        all_comments.extend(data.clean_comments[i])
        vader_comments.extend(data.vader_comments[i])
        flair.extend(data.author_flair[i])

        for j in data['author_flair'][i]:
            if j == data['home_team'][i] or j == data['away_team'][i]:
                fan.append(1)
            else:
                fan.append(0)
    comment_data = list(zip(all_comments, vader_comments, flair, fan))
 
    all_comments = pd.DataFrame(comment_data, columns=[
                                'clean_comments', 'vader_comments', 'flair', 'fan'])
    all_comments.dropna(subset=['clean_comments', 'vader_comments'], inplace=True)

    return all_comments
