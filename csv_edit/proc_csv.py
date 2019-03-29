"""
This module will edit IRAhandle_twwets csv files.
It will remove non-English tweets,
leave only the tweet content and if it's a Troll(Left/Right Troll and Fearmonger)
and normal tweet.
"""
import csv

tweets = list()
propogandaCat = ['RightTroll', 'LeftTroll', 'Fearmonger']

fileList = ['IRAhandle_tweets_' + str(i+1) + '.csv' for i in range(13)]

for cs in fileList:
    with open(cs, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if 'English' in row[4] and 'Unknown' not in row[13]:
                tweets.append([row[2], '1' if row[13] in propogandaCat else '0'])


with open('output.csv', 'wb') as file:
    writer = csv.writer(file)
    writer.writerows(tweets)
