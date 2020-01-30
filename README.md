# Reddit /r/hockey sentiment analysis:

## Problem: 
During a regular-season night in the National Hockey League (NHL), the number 
of games occuring can range from 1 to 15 (31 Teams). On nights with many games
coinciding it is difficult for the viewer to determine which game  is most
likely to be a "good" hockey game. Often times, games with the best teams lead 
to defensively-focused, controlled, low-scorring affairs while other match-ups 
can be offensively-focused, loose, and high-scoring. However, this isn't the 
case all the time and merely guesses which game will display high-quality 
hockey are often wrong. Which game should a fan watch on a given night?
    
## Goal:
Build a classification model to rank the probabilities that NHL regular-season
games on a given night will be a 'good' game based on the participating teams'
recent performance. Provide NHL fans (users) the with nightly rankings to guide
viewing.
    
## Methodology:
The first step to solving this problem is to define what a 'good' hockey game 
is. I have my own opinions about what I like to watch in a hockey game, but the
goal is to build a general model that all NHL fans can use. The step is an 
unsupervised learning problem and observation (NHL regular season game) labels 
will be sourced from the online forum, [Reddit](https://www.reddit.com). The 
website contains different 'subreddits' (sub-forums) correspoding to many 
topics, including [r/hockey](https://www.reddit.com/r/hockey) with 889k members.
Within this subreddit, after every NHL game, a post titled 'Post Game Thread' 
is generated which lists the game information and statistics. Users, often with 
flair (a symbol displaying a team's logo next to the username, denoting fanhood),
publish comments in a comment thread under the post discussing the game that 
just occured. To define what a 'good' hockey game is, the comments will be 
scraped from the Post Game Threads and treated as viewer-reviews of the 
respective NHL game. Using natural language processing (NLP) the reviews will 
undergo sentiment analysis and assigned a  sentiment polarity score indicating 
the positive or negative nature of the comment. The scores will then be 
aggregated among all comments in a Post Game Thread and each game will be 
assigned an overall sentiment polarity score, denoting viewers' reaction to 
the game. What game qualities lead to positive/negative sentiment polarity 
scores?
   
The second step is to predict whether upcoming games will obtain high sentiment
polarity scores, a proxy for game-quality, based on each teams' recent 
performance. Can a model be built to predict game quality (positive sentiment 
polarity scores)?
   
## Results:
Initial investigation into sentiment scoring suggests that close (low goal
differential) and highly offensive (more shots and goals) games are generally
regarded as better hockey games
    
## Future Work:
1. Investigate regression of sentiment scores with recent team performance. Can 
sentiment scores be predicted?
2.  Investigate clustering techniques to label similar games according to game
statistics. Build a classification or reccomendation model.
3. Reddit is only a segment of NHL fanbase, expand to other social media and 
forums

    
## Files:
* Python Scripts:
    * scrape_reddit_posts -> used to scrape www.reddit.com/r/hockey post game
    threads
    * scrape_hockey_reference -> used to scrape game statistics from 
    www.hockey-reference.com
    * clean_data -> used to clean reddit data and merge with hockey reference 
    data
    * clean_text -> used to clean/prepare text for sentiment analysis
    * sentiment_analysis -> used to analyze sentiment of reddit comment text
    * data_viz -> used to produce graphs in Technical Notebook.ipynb
* Technical Notebook.ipynb:
    * Technical breakdown/layout of project
* /data/:
    * all_game_data_YYYY -> hockey-reference game data for YYY(Y-1) to YYYY NHL
    season
    * team_game_data_YYYY -> hockey-reference team-specific data for YYY(Y-1) 
    to YYYY NHL season
    * reddit_YYYY_YYYY -> reddit game threads for YYYY-YYYY season
    * hockey.tsv -> SocialSent /r/hockey lexicon-specific sentiment score 
    dictionary
    * comment_nlp_data -> individual comments with sentiment scores
    * thread_data_nlp -> reddit threads with all data and sentiment scores