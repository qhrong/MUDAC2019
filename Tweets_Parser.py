# This file is used to parse tweet data, including three parts: China Tweets, Farmer Tweets and Soybean Tweets

# Open text files
# Column names: text, created_at, retweet_count, favorite_count, is_retweet, id_str
def file_opener(filename):
    with open(filename, "r") as f1:
        return f1.readlines()[1:]
# Split up string to get tweet content
def str_spliter(filename):
    # This spliter will keep tweet content and time only
    temp_txt = [i.split(',', 1)[1] for i in filename]
    time = [i.rsplit(',', -5)[1] for i in temp_txt]
    time = [i.split(',', 1)[0] for i in time]
    text = [i.rsplit(',', -5)[0] for i in temp_txt]
    return time, text

china = file_opener("China tweets @realDonaldTrump.txt")
farmer = file_opener("FarmerTweets @realDonaldTrump.txt")
soybean = file_opener("soybeans tweets @realDonaldTrump.txt")

time_china, china = str_spliter(china)
time_farmer, farmer = str_spliter(farmer)
time_soybean, soybean = str_spliter(soybean)


# Count of twitter during time range of interest 11/14/2016 ~ 08/30/2019
import pandas as pd
import datetime
import numpy as np
date_ls = time_soybean + time_farmer + time_china

res = []
for i in date_ls:
    try:
        res.append(datetime.datetime.strptime(i, '%m-%d-%Y %H:%M:%S'))
    except:
        continue

date_df = {"Date": res}
date_df = pd.DataFrame(date_df)
date_df['Date'] = [i.date() for i in date_df['Date']]
date_df = date_df.groupby(date_df['Date']).size().reset_index(name='Count')

# Formalize the data format
date_list = pd.DataFrame(pd.date_range(start='11-14-2017', end='08-30-2019', freq='D'))
date_list.columns = ['Date']
date_list['Date'] = [i.date() for i in date_list['Date']]
date_full = pd.merge(date_list, date_df, how='left', on='Date')
date_full = date_full.fillna(0)
date_full.to_csv('Twitter_count.csv', header=True)

# Load in Price data and merge
July = pd.read_excel('ActiveSoybeanContractsforJuly2020.CSV.xlsx', header=3)
May = pd.read_excel('ActiveSoybeanContractsForMay2020.CSV.xlsx', header=3)
March = pd.read_excel('ActiveSoybeanContractsForMarch2020.CSV.xlsx', header=3)

for i in [July, May, March]:
    i['Date'] = [ele.date() for ele in i['Date']]
    tmp_df = pd.merge(i, date_df, on='Date', how='left')
    tmp_df.fillna(0, inplace=True)
    # print(tmp_df)

    # Plot count of tweet vs low ~ high price range
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(tmp_df['Date'], tmp_df['Open'], color="red")
    ax2.bar(tmp_df['Date'], tmp_df['Count'], width=1.5)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Open Price')
    ax2.set_ylabel('Tweet Count')
    ax2.set_ylim(0, 70)

    # plt.show()
    # It seems that tweeting has negative impact on contract prices. And impact's lags on different contract type are different.


# Sentiment analysis using nltk vs Contract Price change
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

#nltk.download('stopwords')
#nltk.download('punkt')


def tweet_filter(tweet):

    # Remove stop words and punctuations
    punct_ls = list(string.punctuation)
    punct_ls.append('”')
    punct_ls.append('“')
    punct_ls.append('...')
    punct_ls.append('..')
    punct_ls.append('``')
    punct_ls.append('.')
    punct_ls.append('/')

    filtered_tweet = [w for w in word_tokenize(tweet) if not w in stopwords.words()]
    filtered_tweet = [w for w in filtered_tweet if not w in punct_ls]

    return filtered_tweet


china_filtered = [tweet_filter(ele) for ele in china]
soybean_filtered = [tweet_filter(ele) for ele in soybean]
farmer_filtered = [tweet_filter(ele) for ele in farmer]
print(china_filtered)

def date_transformer(date):
    res = []
    for i in date:
        try:
            res.append(datetime.datetime.strptime(i, '%m-%d-%Y %H:%M:%S'))
        except:
            continue
    return res


soybean_df = {'Date': [ele.date() for ele in date_transformer(time_soybean)], 'Tweet': soybean_filtered}
soybean_df = pd.DataFrame(soybean_df)
print(soybean_df)
soybean_df.to_csv('soybean_tokens.csv')

farmer_df = {'Date': [ele.date() for ele in date_transformer(time_farmer)], 'Tweet': farmer_filtered}
farmer_df = pd.DataFrame(farmer_df)
print(farmer_df)
farmer_df.to_csv('farmer_tokens.csv')

#china_df = {'Date': [ele.date() for ele in date_transformer(time_china)], 'Tweet': china}
#china_df = pd.DataFrame(china_df)
#print(china_df)



