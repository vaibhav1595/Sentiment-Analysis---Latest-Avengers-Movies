# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:23:05 2020

@author: vaibhav
"""
# Import Libraries

import pandas as pd 
import datetime as dt 
from twitterscraper import query_tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

## Data Collection

# Function for exceptions in text string
def exception_detector(x):
    try:
       return detect(x)
    except:
        None 

# Build Analizer Object
analyzer = SentimentIntensityAnalyzer()

############################################ Avengers: Endgame ############################################

# Before Premier date range
begin_date = dt.date(2019,4,5)
end_date = dt.date(2019,4,25)

# After Premier date range
begin_date_premier = dt.date(2019,4,26)
end_date_premier = dt.date(2019,5,16)

# query tweets with our parameters
tweets_before = query_tweets("#AvengersEndgame", begindate = begin_date, enddate= end_date, limit = 100000)
tweets_after = query_tweets("#AvengersEndgame", begindate = begin_date_premier, enddate = end_date_premier,limit = 100000)

# convert into a dataframe
df_before = pd.DataFrame(t.__dict__ for t in tweets_before)
df_after = pd.DataFrame(t.__dict__ for t in tweets_after)

## Data Cleaning

#filter for english tweets
df_before['lang'] = df_before['text'].apply(lambda x:exception_detector(x))
df_before = df_before[df_before['lang'] == 'en']
df_after['lang'] = df_after['text'].apply(lambda x: exception_detector(x))
df_after = df_after[df_after['lang'] == 'en'] 

#save files
df_before.to_csv('ae_tweets_before_clean.csv')
df_after.to_csv('ae_tweets_after_clean.csv')

# Read Data set
df_before = pd.read_csv('ae_tweets_before_clean.csv')
df_after = pd.read_csv('ae_tweets_after_clean.csv')

#get sentiment scores
sentiment_before = df_before['text'].apply(lambda x: analyzer.polarity_scores(x))
sentiment_after = df_after['text'].apply(lambda x: analyzer.polarity_scores(x))

#put sentiment into dataframe
df_before = pd.concat([df_before, sentiment_before.apply(pd.Series)],1)
df_after = pd.concat([df_after, sentiment_after.apply(pd.Series)],1)

#removed duplicates because of sponsored tweets? 
df_before.drop_duplicates(subset = 'text',inplace = True)
df_after.drop_duplicates(subset = 'text',inplace = True)
df_after['timestamp'] = df_after['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_after = df_after[df_after['timestamp'] > dt.datetime(2019,4,26,0,0,0)]

# Viz for Before Dataframe
df_before['compound'].hist()
df_before['compound'].mean()
df_before['compound'].median()

# Viz for After Dataframe
df_after['compound'].hist()
df_after['compound'].mean()
df_after['compound'].median()

# Ratio (Positive/Negative)
before_ratio = df_before[df_before['compound'] > 0].shape[0] / df_before[df_before['compound'] < 0].shape[0]
after_ratio = df_after[df_after['compound'] > 0].shape[0] / df_after[df_after['compound'] < 0].shape[0]

# Remove zero score
df_before_nz = df_before[df_before['compound'] != 0]
df_after_nz = df_after[df_after['compound'] != 0]

# Viz for Sample
df_before_nz['compound'].sample(100).hist()
df_after_nz['compound'].sample(100).hist()


############################################ Avengers: Infinity War ############################################

# Before Premier date range
a_begin_date = dt.date(2018,4,6)
a_end_date = dt.date(2018,4,26)

# After Premier date range
a_begin_date_premier = dt.date(2018,4,27)
a_end_date_premier = dt.date(2018,5,17)

# query tweets with our parameters
a_tweets_before = query_tweets("#Avengers or #InfinityWar", begindate = a_begin_date, enddate= a_end_date, limit = 100000)
a_tweets_after = query_tweets("#Avengers or #InfinityWar", begindate = a_begin_date_premier, enddate = a_end_date_premier,limit = 100000)

# convert into a dataframe
a_df_before = pd.DataFrame(t.__dict__ for t in a_tweets_before)
a_df_after = pd.DataFrame(t.__dict__ for t in a_tweets_after)

## Data Cleaning

#filter for english tweets
a_df_before['lang'] = a_df_before['text'].apply(lambda x:exception_detector(x))
a_df_before = a_df_before[a_df_before['lang'] == 'en']
a_df_after['lang'] = a_df_after['text'].apply(lambda x: exception_detector(x))
a_df_after = a_df_after[a_df_after['lang'] == 'en'] 

#save files
a_df_before.to_csv('iw_tweets_before_clean.csv')
a_df_after.to_csv('iw_tweets_after_clean.csv')

# Read Data set
a_df_before = pd.read_csv('iw_tweets_before_clean.csv')
a_df_after = pd.read_csv('iw_tweets_after_clean.csv')

#get sentiment scores
a_sentiment_before = a_df_before['text'].apply(lambda x: analyzer.polarity_scores(x))
a_sentiment_after = a_df_after['text'].apply(lambda x: analyzer.polarity_scores(x))

#put sentiment into dataframe
a_df_before = pd.concat([a_df_before, a_sentiment_before.apply(pd.Series)],1)
a_df_after = pd.concat([a_df_after, a_sentiment_after.apply(pd.Series)],1)

#removed duplicates because of sponsored tweets? 
a_df_before.drop_duplicates(subset = 'text',inplace = True)
a_df_after.drop_duplicates(subset = 'text',inplace = True)
a_df_after['timestamp'] = a_df_after['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
a_df_after = a_df_after[a_df_after['timestamp'] > dt.datetime(2018,4,26,0,0,0)]

# Viz for Before Dataframe
a_df_before['compound'].hist()
a_df_before['compound'].mean()
a_df_before['compound'].median()

# Viz for After Dataframe
a_df_after['compound'].hist()
a_df_after['compound'].mean()
a_df_after['compound'].median()

# Ratio (Positive/Negative)
a_before_ratio = a_df_before[a_df_before['compound'] > 0].shape[0] / a_df_before[a_df_before['compound'] < 0].shape[0]
a_after_ratio = a_df_after[a_df_after['compound'] > 0].shape[0] / a_df_after[a_df_after['compound'] < 0].shape[0]

# Remove zero score
a_df_before_nz = a_df_before[a_df_before['compound'] != 0]
a_df_after_nz = a_df_after[a_df_after['compound'] != 0]

# Viz for Sample
a_df_before_nz['compound'].sample(166).hist()
a_df_after_nz['compound'].sample(166).hist()

############################ Comparison between Avengers: Endgame & Avengers: Infinity War ############################

a_df_before_nz['compound'].sample(100).hist()
a_df_after_nz['compound'].sample(100).hist()
df_before_nz['compound'].sample(100).hist()
df_after_nz['compound'].sample(100).hist()

abmean = a_df_before_nz['compound'].mean()
abstd = a_df_before_nz['compound'].std()
a_df_before_nz['compound'].mean()
a_df_after_nz['compound'].mean()
df_before_nz['compound'].mean()
df_after_nz['compound'].mean()

ax1 = sns.distplot(a_df_before_nz['compound'], bins=15, hist = False, label = 'Avengers: Infinity War Before Premier', color = 'r', kde_kws={'linestyle':'--'})
ax2 = sns.distplot(a_df_after_nz['compound'], bins=15, hist = False, label = 'Avengers: Infinity War After Premier',color= 'r')

ax3 = sns.distplot(df_before_nz['compound'], bins=15, hist = False, label = 'Avengers: Endgame Before Premier', color ='blue',  kde_kws={'linestyle':'--'})
ax4 = sns.distplot(df_after_nz['compound'], bins=15, hist = False, label = 'Avengers: Endgame After Premier', color ='blue')
plt.legend()
plt.title('Avengers: Endgame vs Avengers: Infinity War Sentiment')


#samples

# Avengers: Endgame
for i in df_after[df_after['compound'] >= .9]['text'].sample(5):
    print(i)
    print(' ')
    
for i in df_after[df_after['compound'] <= .9]['text'].sample(5):
    print(i)
    print(' ')

# Avengers: Infinity War
for i in a_df_after[a_df_after['compound'] >= .9]['text'].sample(5):
    print(i)
    print(' ')
    
for i in a_df_after[a_df_after['compound'] <= .9]['text'].sample(5):
    print(i)
    print(' ')















