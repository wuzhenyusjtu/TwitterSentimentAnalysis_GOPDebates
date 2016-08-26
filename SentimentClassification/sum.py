import csv

def float_lst(lst):
    lst = [float(x) for x in lst]
    return lst


fpr = open('Sentiment.csv', 'rb')
reader = csv.reader(fpr)
tweets = []
for row in reader:
    tweet = row
    #print tweet
    tweets.append(tweet)
    #label = row[5]
    #training_vects_y.append(get_category(label))
    #training_labels_y.append(label)
fpr.close()
tweets = [float_lst(tweet) for tweet in tweets]
summ = [sum(tweet) for tweet in tweets]
print sum(summ)