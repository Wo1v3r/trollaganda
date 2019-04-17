"""
This module will edit IRAhandle_twwets csv files.
It will remove non-English tweets,
leave only the tweet content and if it's a Troll(Left/Right Troll and Fearmonger)
and normal tweet.
"""
import csv

tweets = list()
count_trolls = {'Troll': 0, 'Not troll: 0}
propogandaCat = ['RightTroll', 'LeftTroll', 'Fearmonger']

fileList = ['IRAhandle_tweets_' + str(i+1) + '.csv' for i in range(13)]

for cs in fileList:
    with open(cs, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if 'English' in row[4] and 'Unknown' not in row[13]:
                if row[13] in propogandaCat:
                    tweets.append([row[2], '1']
                    count_trolls['Troll'] += 1
                else:
                    tweets.append([row[2], '0']
                    count_trolls['Not troll'] += 1  

print(count_trolls)  # {'Troll': 1138095, 'Not troll': 971827}

with open('output.csv', 'wb') as file:
    writer = csv.writer(file)
    writer.writerows(tweets)
