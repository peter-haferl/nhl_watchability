'''
Scrape reddit using reddit API and pushshift reddit API to collect bot-generated post-NHL-game
posts and comment threads as raw json files 

reddit_username, reddit_password = Username and password for reddit account with 
API access
'''

import requests
import pandas as pd
import nltk
import datetime
from time import sleep
from tqdm import tqdm_notebook
import pickle

reddit_username = input()
reddit_password = input()

def total_scrape(start_date, end_date):
    '''
    Get all subcomments and add to df for all reddit posts
    
    Args = start_date, end_date (strings)
    
    Date Format = (YEAR-MONTH-DAY); e.g. 2017-10-15 
    '''
    reddit_posts, d = get_all_reddit_hockey_posts(start_date, end_date)
    reddit_posts = pd.DataFrame(reddit_posts, columns = ['title_json', 'comments_json'])
    reddit_posts = reddit_posts.copy()
    reddit_posts['comments_json'] = reddit_posts['comments_json'].map(lambda x: [x])

    d = get_api_access()
    
    for i in tqdm_notebook(range(0, len(reddit_posts))):
        ind_post_comments = reddit_posts.iloc[i][1][0]['data']['children']
        parent_url = reddit_posts.iloc[i][0]['data']['children'][0]['data']['url']

        sub_comments = get_sub_comments(ind_post_comments=ind_post_comments, parent_url=parent_url, d=d)

        reddit_posts.iloc[i][1].extend(sub_comments)
    return reddit_posts


def get_all_reddit_hockey_posts(start_date, end_date):
    '''
    Output list of tuples containing post name and string of post comments for all Post Game Threads in the 
    2018-2019 Regular NHL Season from the r/hockey/ subreddit
    '''
    links = get_reg_season_links(start_date, end_date)
    d = get_api_access()
    
    reddit_posts = []
    
    # Iterate through all links and pull post comment data
    
    for link in tqdm_notebook(links):
    
        token = 'bearer ' + d['access_token']
        base_url = 'https://oauth.reddit.com'
        headers = {'Authorization': token, 'User-Agent': f'{reddit_username} by {reddit_username}'}
        response = requests.get(base_url + link[22:], headers=headers)

        if response.status_code != 200:
            raise(f'API Error: {response.status_code}')

        post_comments_json = response.json()[1]['data']['children']
        post_comments_string = get_all_comments(post_comments_json)
        title = response.json()[0]['data']['children'][0]['data']['title']
        reddit_posts.append(response.json())
        # Reddit API accepts 60 requests/minute - Limit to one a second
        sleep(1)
    return reddit_posts, d


def get_all_comments(post_comments_json):
    '''
    Output all comments from a reddit post in a single string
    
    comments = response.json()[1]['data']['children'] where response is reddit API response
    '''
    comment_string = ''
    for com in post_comments_json:
        # Beyond 10 layers of comment replies, a new page must be loaded
        if com['kind'] == 'more':
            continue
        if com['data']['depth'] >= 10:
            continue
        comment_string = comment_string + ' ' + com['data']['body']
        if type(com['data']['replies']) == str:
            continue
        else:
             comment_string = comment_string + ' ' + get_all_comments(com['data']['replies']['data']['children'])
    return comment_string


def get_reg_season_links(start_date, end_date):
    '''
    Uses pushshift reddit API to retrieve all Post Game Thread posts in r/hockey/ in the 2018-2019 Regular NHL Season
    Outputs a list of links
    '''
    # Define number of days today is from start of 2018 Reg Season until End
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    today = pd.to_datetime('today')
    days_from_start = (today-start_date).days
    days_from_end = (today-end_date).days
    # Define Start/End Points for API Pagination
    end = days_from_end
    start = days_from_end + 50
    # Paginate through API dates for 2018-2019 Reg Season
    links = []
    while start <= (days_from_start + 50):
        query = f'?title=Post Game Thread&subreddit=hockey&after={start}d&before={end}d&limit=1000'
        resp = requests.get(url = f'https://api.pushshift.io/reddit/search/submission/{query}')
        if resp.status_code != 200:
            raise(f'API Request Failed: {resp.status_code}')
        data = resp.json()
        links.extend([x['full_link'] for x in data['data']])
        end += 50
        start += 50
        sleep(1)
    return links

def get_api_access():
    '''
    Get API Access through Credentials
    
    Output json containing Reddit API access token
    '''
    base_url = 'https://www.reddit.com/'
    data = {'grant_type': 'password', 'username': f'{reddit_username}', 'password': f'{reddit_password}'}
    auth = requests.auth.HTTPBasicAuth('qrPWgz130msLPA', 'j4ON3juJnjLC_c-LqQnEKeaOhzU')
    r = requests.post(base_url + 'api/v1/access_token',
                      data=data,
                      headers={'user-agent': f'{reddit_username} by {reddit_username}'},
                      auth=auth)
    d = r.json()
    return d


def get_sub_comments(ind_post_comments, parent_url, d):
    '''
    Get comments (if any) from post that did not immediately load in page
    
    ind_post_comments: comment list of jsons for an individual post (posts_df.iloc[i][1][0]['data']['children'])
    parent_url: url for individual post (posts_df.iloc[i][0]['data']['children'][0]['data']['url'])
    d: Reddit API access token (generated with get_api_access method)
    '''
    extra_comments = []
    for i in ind_post_comments:
        try:
            if i['kind'] == 'more':
                for com_link in tqdm_notebook(i['data']['children'], leave=False):
                    token = 'bearer ' + d['access_token']
                    base_url = 'https://oauth.reddit.com'
                    link = parent_url[22:] + com_link
                    headers = {'Authorization': token, 'User-Agent': f'{reddit_username} by {reddit_username}'}
                    response = requests.get(base_url + link, headers=headers)

                    if response.status_code != 200:
                        raise(f'API Error: {response.status_code}')

                    post_comments_json = response.json()[1]['data']['children']
                    sleep(1)
                    extra_comments.extend(post_comments_json)
        except:
            continue
    return extra_comments
