######################### Packages #########################
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt



######################### Keys ##############################
#copy and paste the consumer/api public key here
consumerKey=  'XXXXXXXXXX'

#copy and paste the consumer/api secret/private key here
consumerSecret= 'XXXXXXXXXX' 

#copy and paste the public access token here
accessToken=  'XXXXXXXXXX'

#copy and paste the private/secret access token here
accessTokenSecret=  'XXXXXXXXXX'



######################### Authentication ####################
#Authorising Handler
authenticate= tweepy.OAuthHandler(consumerKey, consumerSecret)

#Authenticating Access Token
authenticate.set_access_token(accessToken, accessTokenSecret)

#Establishing API connection
api= tweepy.API(authenticate, wait_on_rate_limit=True)



###################### Extracting Tweets ####################
#Generating request for tweets
tweets= api.user_timeline(screen_name="POLYMERUPDATE", count=100, lang="en", tweet_mode="extended")

#Printing recent five tweets
print("Recent tweets : \n")
i=1
for tweet in tweets[0:5]:
  print(str(i)+") "+tweet.full_text + "\n")
  i+=1



#################### Creating Dataframe #####################
#Creating a pandas dataframe to store all the tweets
df= pd.DataFrame(data= [tweet.full_text for tweet in tweets], columns= ['Tweets'])
df.head()




#################### Cleaning Text ##########################
#function to remove unwanted text
def cleanText(text):
  #removing @ tags
  text= re.sub(r'@[A-Za-z0-9]+',"",text)

  #removing hastags
  text= re.sub(r'#',"",text)

  #removing RT/RTs (ReTweets)
  text= re.sub(r'RT[\s]+',"",text)

  #removing weblink
  text= re.sub(r'https?:\/\/\S+',"",text)

  #removing unwanted or repeating words
  text= re.sub('To read the article, visit:',"",text)
  text= re.sub('Read full article :',"",text)
  text= re.sub('PolymerNews',"",text)
  text= re.sub('polymernews',"",text)
  text= re.sub('USAmarket',"",text)
  text= re.sub('Europemarket',"",text)
  text= re.sub('Watch Now:',"",text)
  text= re.sub('CrudeOil Naphtha',"",text)
  return text

#applying the clean text function to all the tweets
df['Tweets']= df['Tweets'].apply(cleanText)



#################### Subjectivity & Polarity ####################
#function to get the subjectivity of a tweet using textblob package
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

#function to get the polarity of a tweet using textblob package
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

#creating two new column that will store the subjectivity and objectivity of all tweets
df['Subjectivity']= df['Tweets'].apply(getSubjectivity)
df['Polarity']= df['Tweets'].apply(getPolarity)



######################### Word Cloud #########################
#joining all the tweets to create a text corpus
allWords= ''.join([twt for twt in df['Tweets']])

#generating the word cloud using the WordCloud package
wordCloud= WordCloud(random_state=20, max_font_size=120).generate(allWords)

#plotting the text cloud
plt.figure(figsize=(10,5))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()



#################### Analysis Sentiment #########################
#function to get the sentiment of a tweet using the polarity of that tweet
def getAnalysis(score):
  if score<0:
    return "Negative"
  elif score==0:
    return "Neutral"
  else:
    return "Positive"

#creating a new column in the dataframe which will store whether a tweet is positive, neutral or negative
df['Analysis']= df['Polarity'].apply(getAnalysis)
df.head(10)



############### Printing Positive Tweets ####################
#printing all the positive tweets; most positive tweets being on top
j=1
sortedDF= df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):
  if sortedDF['Analysis'][i]== 'Positive':
    print(f"{j}) {sortedDF['Tweets'][i]}".replace(".","").strip())
    print()
    j+=1



############### Printing Negative Tweets ####################
#printing all the negative tweets in order of negativity
j=1
sortedDF= df.sort_values(by=['Polarity'],ascending=False)
for i in range(0, sortedDF.shape[0]):
  if sortedDF['Analysis'][i]== 'Negative':
    print(f"{j}) {sortedDF['Tweets'][i]}".replace(".","").strip())
    print()
    j+=1



######################### Visuals ############################
#scatter plot of subjectivity and polarity of the tweets
plt.figure(figsize=(10,5))

plt.scatter(df['Polarity'],df['Subjectivity'], color='#5f9ea0' )

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')

plt.grid(axis = 'x')

plt.axhline(y = 0.5, color = 'lightcoral', linestyle = '--')
plt.axvline(x = 0, color = 'darkred', linestyle = '--')

plt.show()

#Histogram representing number of positive, neutral, and negative tweets
plt.figure(figsize=(10,5))

plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')

df['Analysis'].value_counts().plot(kind='bar', color=['green', 'lavender', 'maroon'])

plt.show()