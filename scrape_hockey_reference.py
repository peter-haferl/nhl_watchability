'''
Scrapes hockey reference for game data and team-specific game data
    -Use scrape_game_data() for game data
    -Use scrape_team_data() for team-specific game data
'''

import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm_notebook

team_name = ['San Jose Sharks',
             'Toronto Maple Leafs',
             'Vancouver Canucks',
             'Washington Capitals',
             'Buffalo Sabres',
             'Carolina Hurricanes',
             'Colorado Avalanche',
             'Dallas Stars',
             'Detroit Red Wings',
             'New York Rangers',
             'Ottawa Senators',
             'Pittsburgh Penguins',
             'St. Louis Blues',
             'Vegas Golden Knights',
             'Columbus Blue Jackets',
             'Los Angeles Kings',
             'Arizona Coyotes',
             'Calgary Flames',
             'Minnesota Wild',
             'New Jersey Devils',
             'New York Islanders',
             'Tampa Bay Lightning',
             'Chicago Blackhawks',
             'Anaheim Ducks',
             'Boston Bruins',
             'Nashville Predators',
             'Philadelphia Flyers',
             'Winnipeg Jets',
             'Florida Panthers',
             'Montreal Canadiens',
             'Edmonton Oilers']

team_code = ['SJS',
            'TOR',
            'VAN',
            'WSH',
            'BUF',
            'CAR',
            'COL',
            'DAL',
            'DET',
            'NYR',
            'OTT',
            'PIT',
            'STL',
            'VEG',
            'CBJ',
            'LAK',
            'ARI',
            'CGY',
            'MIN',
            'NJD',
            'NYI',
            'TBL',
            'CHI',
            'ANA',
            'BOS',
            'NSH',
            'PHI',
            'WPG',
            'FLA',
            'MTL',
            'EDM']

name_to_code = dict(zip(team_name, team_code))


def scrape_team_data(years):
    '''
    Scrape all team stats for specified years from https://www.hockey-reference.com/
    
    Inputs:
    
    years -> An iterable of years to scrape
    
    Outputs:
    
    Saves a pickle for each year containing:
    
        -list of tuples 
        -tuple[0] is team 3-letter code 
        -tuple[1] is a pandas dataframe with team stats
    '''
    for year in years:
        teams = ['ANA',
                'ARI',
                'BOS',
                'BUF',
                'CAR',
                'CGY',
                'CHI',
                'CBJ',
                'COL',
                'DAL',
                'DET',
                'EDM',
                'FLA',
                'LAK',
                'MIN',
                'MTL',
                'NSH',
                'NJD',
                'NYI',
                'NYR',
                'OTT',
                'PHI',
                'ARI',
                'PIT',
                'SJS',
                'STL',
                'TBL',
                'VAN',
                'TOR',
                'VEG',
                'WPG',
                'WSH']
        # Vegas Golden Knight Entered NHL in 2017-2018 Season
        if year < 2017:
            teams.remove('VEG')
            
        team_data = []
        for team in tqdm_notebook(teams):
            url = f'https://www.hockey-reference.com/teams/{team}/{year}_gamelog.html'
            headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0'}
            resp2 = requests.get(url=url, headers=headers)
            soup = BeautifulSoup(resp2.text, features='lxml')
            table = soup.find('table')
            table_body = table.find('tbody')
            rows = table_body.find_all('tr')
            data= []
            for row in rows:
                cols = row.find_all('td')
                games_played = row.find('th')
                cols = [ele.text.strip() for ele in cols]
                cols.append(games_played.text)
                data.append(cols)
            team_game_data_df = pd.DataFrame(data)
            team_game_data_df.columns = ['date',
                                    '?', 
                                    'opponent', 
                                    'goals_for', 
                                    'goals_against', 
                                    'win/loss', 
                                    'ot/so',
                                    '?',
                                    'team_sog', 
                                    'team_pim',
                                    'team_pp_goal',
                                    'team_pp_oppurtunities',
                                    'team_shorthanded_goals',
                                    '?',
                                    'opponent_sog', 
                                    'opponent_pim',
                                    'opponent_pp_goal',
                                    'opponent_pp_oppurtunities',
                                    'opponent_shorthanded_goals',
                                    '?',
                                    'corsi_for',
                                    'corsi_against',
                                    'corsi_for%',
                                    'fenwick_for',
                                    'fenwick_against',
                                    'fenwick_for%',
                                    'faceoff_wins',
                                    'faceoff_loses',
                                    'faceoff%',
                                    'off_zone_start%',
                                    'pdo',
                                    'games_played'
                                    ]
            team_game_data_df.drop(columns=['?'], inplace=True)
            team_game_data_df.dropna(inplace=True)
            team_game_data_df['date'] = team_game_data_df['date'].map(lambda x: pd.to_datetime(x))
            team_game_data_df['opponent'] = team_game_data_df['opponent'].map(lambda x: name_to_code[x])
            team_data.append((team, team_game_data_df))
            time.sleep(2)
        pd.to_pickle(team_data, f'./data/team_game_data_{year}')

        
def scrape_game_data(years):
    '''
    Scrape all game stats for specified years from https://www.hockey-reference.com/
    
    Inputs:
    
    years -> An iterable of years to scrape
    
    Outputs:
    
    Pickles a dataframe for each year
    '''
    for year in years:
        url = f'https://www.hockey-reference.com/leagues/NHL_{year}_games.html'
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0'}
        resp = requests.get(url=url, headers=headers)
        soup = BeautifulSoup(resp.text, features='lxml')
        table = soup.find('table')
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        
        data= []
        for row in rows:
            cols = row.find_all('td')
            date = row.find('th')
            cols = [ele.text.strip() for ele in cols]
            cols.append(date.text)
            data.append(cols)

        game_data_df = pd.DataFrame(data)
        game_data_df.columns = ['visitor', 'visitor_goals', 'home',
                                'home_goals', 'ot/so', 'attendance',
                                'len_game', 'notes', 'date']
        game_data_df.drop(columns=['notes'], inplace=True)
        game_data_df['visitor'] = game_data_df['visitor'].map(lambda x: name_to_code[x])
        game_data_df['home'] = game_data_df['home'].map(lambda x: name_to_code[x])
        game_data_df['date'] = game_data_df['date'].map(lambda x: pd.to_datetime(x))
        
        pd.to_pickle(game_data_df, f'./data/all_game_data_{year}')
