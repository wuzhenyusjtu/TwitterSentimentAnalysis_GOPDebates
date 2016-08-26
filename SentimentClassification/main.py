from Preprocessor import Preprocessor
from FeatureExtractor import FeatureExtractor
import csv
import nltk.classify
from NaiveBayesClassifier import NaiveBayesClassifier
from MaxEntropyClassifier import MaxEntropyClassifier
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

fpr = open('Sentiment.csv', 'rb')
reader = csv.reader(fpr)
next(reader, None)

tweets = []
training_labels_y = []
training_vects_y = []
for row in reader:
    tweet = row[15]
    #print tweet
    tweets.append(tweet)
    label = row[5]
    training_vects_y.append(get_category(label))
    training_labels_y.append(label)
fpr.close()


preprocessor = Preprocessor('stopWords.txt')
feature_extractor = FeatureExtractor(usingNegation = True, usingStemming = True, usingCorrection = False)


training_dicts_x = []
training_vects_x = []
training_x = []
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
    #feature_vect = feature_extractor.build_feature_vector(features)
    feature_dict = feature_extractor.build_feature_dict(features)
    #training_vects_x.append(feature_vect)
    training_dicts_x.append(feature_dict)
    #print '##################################################'
    count += 1
    if count % 100 == 0:
        print count

'''
with open('raw_validation_vects.csv', 'wb') as fp:
    writer = csv.writer(fp)
    # write world list into '.csv' file
            # write feature number matrix into '.csv' file
    for row in training_vects_x:
        writer.writerow(row)
'''

'''
with open('stem_training_vects_x.pkl','wb') as handle:
    cPickle.dump(training_vects_x, handle)
'''

#with open('raw_training_dicts_x.pkl','wb') as handle:
#    cPickle.dump(training_dicts_x, handle)

'''
with open('training_vects_y.pkl','wb') as handle:
    cPickle.dump(training_vects_y, handle)
'''
'''
NB_classifier = NaiveBayesClassifier(needTraining = True)
#training_set = feature_extractor.build_training_set(training_x, training_labels_y)
#print type(training_set)
NB_classifier.train(training_dicts_x, training_labels_y)
print 'TRAINING COMPLETED!!!'
#print NB_classifier.test(training_dicts_x[0])
print NB_classifier.accuracy(NB_classifier.test(training_dicts_x), training_labels_y)
NB_classifier.get_most_inform_features(100)
'''

ME_classifier = MaxEntropyClassifier(needTraining = True)
#training_set = feature_extractor.build_training_set(training_x, training_labels_y)
#print type(training_set)
ME_classifier.train(training_dicts_x, training_labels_y)
print 'TRAINING COMPLETED!!!'
#print NB_classifier.test(training_dicts_x[0])
print ME_classifier.accuracy(ME_classifier.test(training_dicts_x), training_labels_y)
ME_classifier.get_most_inform_features(100)
