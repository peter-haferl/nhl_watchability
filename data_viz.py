'''
Functions for visualizing data for EDA
'''
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from sklearn.preprocessing import MinMaxScaler

# game_data = ['num_comments', 'num_fights', 'attendance', 'len_game', 'corsi_for%',
#              'fenwick_for%', 'faceoff%', 'off_zone_start%', 'pdo', 'total_sog',
#              'total_pp_goals', 'total_goals', 'total_pp_opportunity', 'total_sh_goals',
#              'sog_diff', 'pp_goals_diff', 'goal_diff', 'pp_opportunity_diff', 'sh_goals_diff',
#              'sog_norm_diff', 'pp_goals_norm_diff', 'goal_norm_diff',
#              'pp_opportunity_norm_diff','sh_goals_norm_diff', 'ot/so', 'comeback?', 'win/loss']

game_data = ['num_comments', 'num_fights', 'attendance', 'len_game', 'corsi_for%',
             'fenwick_for%', 'faceoff%', 'off_zone_start%', 'pdo', 'total_sog',
             'total_pp_goals', 'total_goals', 'total_pp_opportunity', 'total_sh_goals',
             'sog_diff', 'pp_goals_diff', 'goal_diff', 'pp_opportunity_diff', 'sh_goals_diff',
             'ot/so', 'comeback?', 'win/loss']


def plot_team_effect_binary_data(data, statistic=None, game_venue='home'):
    '''Plot Means of binary data stats ('win/loss', 'comeback?')
    '''
    if game_venue == 'home':
        home_data = data.groupby(by='home_team').mean()[statistic]
        plt.figure(figsize=(25,10))
        sns.scatterplot(x='home_team', y=statistic, data=home_data.reset_index(drop=False),
                        s=400, marker='s', color='darkblue')
        plt.hlines(y=home_data.mean(), xmin=-1, xmax=30.4, colors='red', 
                   label='Population Mean', linewidth=2)
        
    elif game_venue == 'away':
        away_data = data.groupby(by='away_team').mean()[statistic]
        plt.figure(figsize=(25,10))
        sns.scatterplot(x='away_team', y=statistic, data=away_data.reset_index(drop=False),
                        s=400, marker='s', color='darkblue')
        plt.hlines(y = away_data.mean(), xmin=-1, xmax=30.4, colors='red',
                   label='Population Mean', linewidth=2)
        
    elif game_venue == 'all':
        home_data = data.groupby(by='home_team').mean()[statistic]
        away_data = data.groupby(by='away_team').mean()[statistic]
        if statistic == 'win/loss':
            away_data = away_data.map(lambda x: 1-x)
        all_data = (home_data + away_data)/2
        all_data.rename_axis('team', inplace=True)
        plt.figure(figsize=(25,10))
        sns.scatterplot(x='team', y=statistic, data=all_data.reset_index(drop=False),
                        s=400, marker='s', color='darkblue')
        plt.hlines(y = all_data.mean(), xmin=-1, xmax=30.4, colors='red',
                   label='Population Mean', linewidth=2)
        
    title_font = {'size':'22',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'20',
                 'weight': 'medium'}
    tick_font = {'size': '16',
                 'weight': 'medium'}
    statistic = statistic.replace('_', ' ').upper()
    plt.title(f'Team Effect on {statistic} in {game_venue.title()} Games', **title_font)
    plt.ylabel(f'{statistic}', **axis_font)
    plt.xlabel('Team', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font);

# def plot_team_effect_binary_data(data, statistic=None, game_venue='home'):
#     '''Plot Means of binary data stats ('win/loss', 'comeback?')
#     '''
#     if game_venue == 'home':
#         home_data = data.groupby(by='home_team').mean()[statistic]
#         plt.figure(figsize=(25,10))
#         sns.scatterplot(x='home_team', y=statistic, data=home_data.reset_index(drop=False),
#                         s=400, marker='s', color='darkblue')
#         plt.hlines(y=home_data.mean(), xmin=-1, xmax=30.4, colors='red', 
#                    label='Population Mean', linewidth=2)
        
#     elif game_venue == 'away':
#         away_data = data.groupby(by='away_team').mean()[statistic]
#         plt.figure(figsize=(25,10))
#         sns.scatterplot(x='away_team', y=statistic, data=away_data.reset_index(drop=False),
#                         s=400, marker='s', color='darkblue')
#         plt.hlines(y = away_data.mean(), xmin=-1, xmax=30.4, colors='red',
#                    label='Population Mean', linewidth=2)
        
#     title_font = {'size':'22',
#                   'color':'black',
#                   'weight':'medium',
#                   'verticalalignment':'bottom'} 
#     axis_font = {'size':'20',
#                  'weight': 'medium'}
#     tick_font = {'size': '16',
#                  'weight': 'medium'}
#     statistic = statistic.replace('_', ' ').upper()
#     plt.title(f'Team Effect on {statistic} in {game_venue.title()} Games', **title_font)
#     plt.ylabel(f'{statistic}', **axis_font)
#     plt.xlabel('Team', **axis_font)
#     plt.xticks(**tick_font)
#     plt.yticks(**tick_font);

def plot_team_effect(data, statistic=None, game_venue='all', kind='boxplot'):
    '''Plot distribution of game statistics in which a given team plays
    
    statistic = string of game statistic (e.g total_goals, sog_diff). Default = None.
    
    game_venue = 'home' or 'away' or 'all'. Selects venue to show game stats from (i.e.
    'home' shows game stats for all home games of a given team)
    
    kind = 'boxplot' or 'violinplot' or 'stripplot'. Selects which type of distribution 
    plot to display.
    '''  

    if game_venue == 'all':
        teams = sorted(list(data.home_team.unique()))
        team_data = dict()

        for team in teams:
            one_team_data = data.loc[(data.home_team == team) | (data.away_team == team)].reset_index(drop=True)
            team_data[team] = one_team_data[statistic]

        team_data = pd.DataFrame(team_data)

        if kind == 'boxplot':
            plt.figure(figsize=(25,10))
            sns.boxplot(data=team_data)
            plt.hlines(y=data[statistic].median(), xmin=-1, xmax=30.4, colors='red', 
                        label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)
            
        elif kind == 'violinplot':
            plt.figure(figsize=(25,10))
            sns.violinplot(data=team_data)
            plt.hlines(y=data[statistic].median(), xmin=-1, xmax=30.4, colors='red', 
                        label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)

        elif kind == 'stripplot':
            plt.figure(figsize=(25,10))
            sns.stripplot(data=team_data, jitter=0.35)
            plt.hlines(y=data[statistic].median(), xmin=-1, xmax=30.4, colors='red', 
                        label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)

        else:
            return KeyError('Invalid Plot Kind')
        
    elif game_venue == 'home' or 'away':
        
        if kind == 'boxplot':
            plt.figure(figsize=(25,10))
            sns.boxplot(x=f'{game_venue}_team', y=statistic,
                        data=data.sort_values(by=f'{game_venue}_team'))
            plt.hlines(y=data[statistic].median(), xmin=-1, xmax=30.4, colors='red', 
                            label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)
            
        elif kind == 'violinplot':
            plt.figure(figsize=(25,10))
            sns.violinplot(x=f'{game_venue}_team', y=statistic,
                           data=data.sort_values(by=f'{game_venue}_team'))
            plt.hlines(y=data[statistic].mean(), xmin=-1, xmax=30.4, colors='red', 
                            label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)
        
        elif kind == 'stripplot':
            plt.figure(figsize=(25,10))
            sns.stripplot(x=f'{game_venue}_team', y=statistic, jitter=0.35, 
                         data=data.sort_values(by=f'{game_venue}_team'))
            plt.hlines(y=data[statistic].median(), xmin=-1, xmax=30.4, colors='red', 
                            label='Population Median', linewidth=2)
            plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)
            
        else:
            return KeyError('Invalid Plot Kind')
        
    else:
        return KeyError('Invalid game_venue')
    
    title_font = {'size':'22',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'20',
                 'weight': 'medium'}
    tick_font = {'size': '16',
                 'weight': 'medium'}
    sns.despine(offset=0, trim=True)
    statistic = statistic.replace('_', ' ').upper()
    plt.title(f'Team Effect on {statistic} in {game_venue.title()} Games', **title_font)
    plt.ylabel(f'{statistic}', **axis_font, labelpad=5)
    plt.xlabel('Team', **axis_font, labelpad=5)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font);
    
def plot_team_specific_stats(data, statistic=None, against=False):
    '''Plot distribution of team-specific statistics
    
    statistic = string of statistic that is available for both home and away (e.g corsi,
    goals, sog, etc). Default = None.
    
    against = Boolean. False will show distribution of statistic for a given team, True will
    show distribution of statistic against a given team. Default = False.
    ''' 
    teams = list(data.home_team.unique())
    team_data = dict()

    for team in teams:
        if against == False:
            home_statistic = data.loc[(data.home_team == team)]['home_' + statistic]
            away_statistic = data.loc[(data.away_team == team)]['away_' + statistic]
        elif against == True:
            home_statistic = data.loc[(data.home_team == team)]['away_' + statistic]
            away_statistic = data.loc[(data.away_team == team)]['home_' + statistic]

        statistic_data = pd.concat([home_statistic, away_statistic], ignore_index=True)
        team_data[team] = statistic_data
        
    team_data = pd.DataFrame(team_data)
    plt.figure(figsize=(25,10))
    sns.boxplot(data=team_data)
    population_stat_data = (list(data['home_' + statistic]) + 
                            (list(data['away_' + statistic])))
    plt.hlines(y=np.median(population_stat_data), xmin=-1, xmax=30.4,
               colors='red', label='Population Median', linewidth=2)
    plt.legend(loc=(0,0.95), fontsize=16, edgecolor=None)
    title_font = {'size':'22',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'20', 'weight': 'medium'}
    tick_font = {'size': '16', 'weight': 'medium'}
    sns.despine(offset=0, trim=True)
    
    if against == False:
        statistic = statistic + '_for'
    elif against == True:
        statistic = statistic + '_against'
        
    statistic = statistic.replace('_', ' ').upper()
    plt.title(f'Team Effect on {statistic} in Games', **title_font)
    plt.ylabel(f'{statistic}', **axis_font, labelpad=5)
    plt.xlabel('Team', **axis_font, labelpad=5)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font);
    
def plot_numeric_features_dist(data, kind='boxplot', size=(12,8), normalize=True):
    '''
    Plot distributions of numeric features within a DataFrame
    
    size = tuple with desired size of output figure (width, height)
    
    normalize = Boolean (Default = True). If True, numerical data is normalized using
    sklearn.preprocessing.MinMaxScaler(). If False, numerical data is not altered.
    '''
    columns = []
    for column in data.columns:
        if (data[column].dtype != 'O'):
            columns.append(column)
    if normalize == True:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[columns])
        data = pd.DataFrame(scaled_data, columns=columns)
    if kind == 'boxplot':
        plt.figure(figsize=(size))
        sns.boxplot(data=data[columns])
    elif kind == 'violinplot':
        plt.figure(figsize=(size))
        sns.violinplot(data=data[columns], scale='count', cut=0)
    elif kind == 'stripplot':
        plt.figure(figsize=(size))
        sns.stripplot(data=data[columns], jitter=0.25)
    title_font = {'size':'18',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'}
    tick_font = {'size': '12',
                 'weight': 'medium'}
    if normalize == True:
        plt.title(f'Normalized Distribution of Numeric Features', **title_font)
    elif normalize == False:
        plt.title(f'Distribution of Numeric Features', **title_font)
    formatted_x_ticks = [x.replace('_', ' ').replace(' polarity', '').title() for x in columns]
    plt.yticks(**tick_font)
    plt.xticks(ticks=plt.xticks()[0], labels=formatted_x_ticks, rotation=60, ha='right', **tick_font);
    

def plot_game_binary_data(data, size=(16, 5)):
    '''
    Plot distributions of binary/ordinal game data
    '''
    
    title_font = {'size':'16',
              'color':'black',
              'weight':'medium',
              'verticalalignment':'bottom'} 
    axis_font = {'size':'14', 'weight': 'medium'}
    tick_font = {'size': '16', 'weight': 'medium'}
    
    fig, axs = plt.subplots(1, len(data.columns), figsize=size)
    for i, column in enumerate(data.columns):
        axs[i].hist(data[column])
        column = column.upper()
        axs[i].set_title(column, **title_font)
    axs[0].set_ylabel('Frequency', **axis_font);

    
def plot_team_effect_comment_fandom(comment_data, size=(12,6)):
    '''
    Plots percentage of comments from a team fanbase that are in game threads that their respective team played in
    
    comment_data = output of clean_data.extract_comment_data()
    '''
    nhl_abbreviations = ['CBJ', 'TBL', 'MTL', 'FLA', 'NSH', 'SJS', 'LAK', 'BOS', 'COL', 'OTT', 'DAL', 'CGY', 'MIN', 
                    'NYI', 'NYR', 'BUF', 'WSH', 'PIT', 'WPG', 'VAN', 'PHI', 'CHI', 'CAR', 'ANA', 'EDM', 'STL',
                    'DET', 'TOR', 'NJD', 'ARI', 'VGK']
    
    title_font = {'size':'20',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'16', 'weight': 'medium'}
    tick_font = {'size': '14', 'weight': 'medium'}
    
    plt.figure(figsize=size)
    comment_data.groupby('flair').mean().loc[nhl_abbreviations]['fan'].plot(kind='bar')
    plt.hlines(y=np.mean(comment_data.groupby('flair').mean().loc[nhl_abbreviations]['fan']), xmin=0, xmax=31, color='red')
    plt.title(label='Percentage of Fan Comments by Team', **title_font)
    plt.xlabel('Team', **axis_font)
    plt.ylabel('Percetage of Fan Comments', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font);
    
def plot_scoring_dist(comment_data, scoring=None, size=(10, 6)):
    '''
    Plot distributions of binary/ordinal game data
    '''
    perc_neutral = np.round(len(comment_data.loc[comment_data[scoring] == 0])/len(comment_data), 
                            decimals=3)
    title_font = {'size':'16',
              'color':'black',
              'weight':'medium',
              'verticalalignment':'bottom'} 
    axis_font = {'size':'14', 'weight': 'medium'}
    tick_font = {'size': '12', 'weight': 'medium'}
    
    plt.figure(figsize=size)
    comment_data[scoring].plot(kind='hist', bins=100)
    plt.ylabel('Frequency', **axis_font)
    scoring = scoring.replace('_', ' ').title()
    plt.xlabel(f'{scoring}', **axis_font)
    plt.xticks(**tick_font)
    plt.yticks(**tick_font)
    plt.show();
    
    print(f'Percentage of neutral comments: {perc_neutral}')

def plot_nlp_heatmap(thread_data, size=(14,17)):
    '''
    Plot heatmap showing correlation between sentiment scores and game features
    '''
    nlp_data = ['fan%', 'vader_comment',
       'socialsent_comment', 'vader_context',
       'socialsent_context', 'positive_context', 'non-fan_context',
       'no_ref_context']

    corr_data = thread_data[game_data + nlp_data].corr().applymap(lambda x: np.round(x, 2))
    corr_data = corr_data.drop(index=nlp_data)[nlp_data]

    color_bar = {
        'shrink': 0.5,
        'pad': 0.01
    }
            
    title_font = {'size':'20',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'20',
                 'weight': 'medium'}
    tick_font = {'size': '12',
                 'weight': 'medium'}

    plt.figure(figsize=size)
    sns.heatmap(corr_data, annot=True, linewidths=0.1, 
                square=False, cbar=True, cbar_kws=color_bar,
               vmin=-1, vmax=1, cmap="RdBu_r");
    plt.xticks(rotation=30, ha='right', **tick_font)
    plt.yticks(rotation=0, ha='right', **tick_font)
    
    plt.title('Correlation of Game Features with Sentiment Scores', **title_font);
    
def plot_game_feature_heatmap(total_data):
    '''
    Plot showing correlation between game features
    '''    
    corr_data = total_data[game_data].corr().applymap(lambda x: np.round(x, 2))

    mask = np.zeros_like(corr_data)
    mask[np.triu_indices_from(mask)] = True

    color_bar = {
        'shrink': 0.5,
        'pad': 0.01
    }

    title_font = {'size':'20',
                  'color':'black',
                  'weight':'medium',
                  'verticalalignment':'bottom'} 
    axis_font = {'size':'20',
                 'weight': 'medium'}
    tick_font = {'size': '12',
                 'weight': 'medium'}

    plt.figure(figsize=(17,17))
    sns.heatmap(corr_data, annot=True, linewidths=0.1, 
                square=False, cbar=True, cbar_kws=color_bar,
               vmin=-1, vmax=1, cmap="RdBu_r", mask=mask);
    plt.xticks(rotation=90, ha='right', **tick_font)
    plt.yticks(rotation=0, ha='right', **tick_font)

    plt.title('Correlation of Game Features', **title_font);