from Preprocessor import Preprocessor
from FeatureExtractor import FeatureExtractor
import csv
import nltk.classify
import cPickle

def get_category(attribute):
    if attribute == "Positive":
        return ["1", "0", "0"]
    elif attribute == "Negative":
        return ["0", "1", "0"]
    elif attribute == "Neutral":
        return ["0", "0", "1"]
    else:
        # raise ValueError('Attribute is wrong')
        print attribute
        raise ValueError('Attribute is wrong')

def load_tweets_csv(file):
    fpr = open(file, 'rb')
    reader = csv.reader(fpr)
    tweets = []
    training_labels_y = []
    training_vects_y = []
    for row in reader:
        tweet = row[1]
        tweets.append(tweet)
        #label = row[5]
        #training_vects_y.append(get_category(label))
        #training_labels_y.append(label)
    fpr.close()
    return tweets

def load_tweets_pkl(file):
    fp = open(file, 'rb')
    tweets = cPickle.load(fp)
    return tweets

def dump_vects_csv(file, feature_vect_lst):
    with open(file, 'wb') as fp:
        writer = csv.writer(fp)
        # write world list into '.csv' file
        #write feature number matrix into '.csv' file
        for vect in feature_vect_lst:
            writer.writerow(vect)

def dump_vects_pkl(file, feature_vect_lst):
    with open(file,'wb') as handle:
        cPickle.dump(feature_vect_lst, handle)

def load_vects_csv(file):
    fpr = open(file, 'rb')
    reader = csv.reader(fpr)
    feature_vect_lst = []
    for row in reader:
        vect = row
        feature_vect_lst.append(vect)
    fpr.close()
    return feature_vect_lst

def load_vects_pkl(file):
    fp = open(file, 'rb')
    feature_vect_lst = cPickle.load(fp)
    return feature_vect_lst

def get_feature_vect_lst(tweets):
    feature_dict_lst = []
    feature_vect_lst = []
    count = 0
    for tweet in tweets:
        #print tweet
        #print '***********************'
        processed_tweet = preprocessor.preprocess(tweet)
        #print processed_tweet
        #print '***********************'
        features = feature_extractor.extract_features_testing(processed_tweet)
        #print features
        #training_x.append(features)
        feature_vect = feature_extractor.build_feature_vector(features)
        #feature_dict = feature_extractor.build_feature_dict(features)
        feature_vect_lst.append(feature_vect)
        #training_dict_lst.append(feature_dict)
        #print '##################################################'
        count += 1
        if count % 100 == 0:
            print count
    return feature_vect_lst

ted_list = ['ted cruz20160412Tweets.csv','ted cruz20160413Tweets.csv', 'ted cruz20160414Tweets.csv', 'ted cruz20160415Tweets.csv', 'ted cruz20160416Tweets.csv', 'ted cruz20160417Tweets.csv', 'ted cruz20160418Tweets.csv']
trump_list = ['trump20160412Tweets.csv', 'trump20160413Tweets.csv', 'trump20160414Tweets.csv', 'trump20160415Tweets.csv', 'trump20160416Tweets.csv', 'trump20160417Tweets.csv', 'trump20160418Tweets.csv']

preprocessor = Preprocessor('stopWords.txt')
feature_extractor = FeatureExtractor(usingNegation = False, usingStemming = False, usingCorrection = True)
'''
for ted in ted_list:
    tweets = load_tweets_csv(ted)
    feature_vect_lst = get_feature_vect_lst(tweets)
    dump_vects_csv(ted[:-9]+'Vects.csv', feature_vect_lst)
'''
for trump in trump_list:
    tweets = load_tweets_csv(trump)
    feature_vect_lst = get_feature_vect_lst(tweets)
    dump_vects_csv(trump[:-9]+'Vects.csv', feature_vect_lst)