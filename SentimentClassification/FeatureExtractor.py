import re
import csv
from MisspellingCorrector import MisspellingCorrector
import enchant
import nltk.classify
from nltk.stem.wordnet import WordNetLemmatizer
import cPickle

class FeatureExtractor(object):
    def __init__(self, usingCorrection = False, usingNegation = False, usingStemming = False):
        self.usingCorrection = usingCorrection
        self.usingNegation = usingNegation
        self.usingStemming = usingStemming
        self.wnl = WordNetLemmatizer()
        if not usingNegation and not usingStemming and not usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/raw_featurespace_11414.pkl')
            
        elif not usingNegation and not usingStemming and usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/raw_featurespace_11414.pkl')
            self.corrector = MisspellingCorrector('correctionpairs.txt', 'dictionary/raw_dictionary_11414.pkl')
            
        elif not usingNegation and usingStemming and not usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/stem_featurespace_9435.pkl')
    
        elif not usingNegation and usingStemming and usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/stem_featurespace_9435.pkl')
            self.corrector = MisspellingCorrector('correctionpairs.txt','dictionary/stem_dictionary_9435.pkl')
            
        elif usingNegation and not usingStemming and not usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/neg_featurespace_13239.pkl')
            
        elif usingNegation and not usingStemming and usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/neg_featurespace_13239.pkl')
            self.corrector = MisspellingCorrector('correctionpairs.txt', 'dictionary/raw_dictionary_11414.pkl')
            
        elif usingNegation and usingStemming and not usingCorrection:
            self.feature_space = self.get_feature_space('featurespace/stem_neg_featurespace_11071.pkl')
            
        else:
            self.feature_space = self.get_feature_space('featurespace/stem_neg_featurespace_11071.pkl')
            self.corrector = MisspellingCorrector('correctionpairs.txt', 'dictionary/stem_dictionary_9435.pkl')
            
    def get_feature_space(self,fileName):
        feature_space = cPickle.load(open(fileName,'rb'))
        return sorted(feature_space)

    # negation is to add "neg at the end of the word"
    def negation(self, tweet):
        neg = " "

        # find the substring betweetn Negation word and Clause-level punctuation 
        pattern = re.compile(
            r"(.*(never| no|nothing|nowhere|noone| non|none|nobody| not|\
                |havent|hasnt|hadnt|cant|couldnt|cannot|shouldnt|wasnt|\
                |wont|wouldnt|dont|doesnt|didnt|isnt|arent|\
                |n't|aint))([^.:;!?]*)([.:;!?])(.*)")
        m = re.search(pattern, tweet)
        
        if m:
            # add "neg " at the end of a word
            words_in_scope =  m.group(3).split()
            for w in words_in_scope:
                neg += w + "neg "

                # restore tweets
            tweet = m.group(1) + neg + m.group(4) + " " + m.group(5)

        return tweet.split()

    # stemming according to WordNet
    def stem_wordnet(self, word):

        # obtain the word class
        tag = nltk.pos_tag(nltk.word_tokenize(word))

        # word class for verb can be different, but the first two letters must be "VB"
        if len(tag[0][1]) >= 2 and (tag[0][1])[0:2] == 'VB':
            return self.wnl.lemmatize(word, 'v')

        else:
            return self.wnl.lemmatize(word)
            
    # strip punctuations
    def strip_punct(self, words):
        tmp = []
        for w in words:
            # strip symbols in a word
            w = re.sub(r'[\'"?:*,./!]+','',w)
            # w = w.replace("'s", '')
            if w != '':
                tmp.append(w)

        return tmp
        
    def extract_features_testing(self, words):
        if not self.usingNegation and not self.usingStemming and not self.usingCorrection:
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            return words
            
        elif not self.usingNegation and not self.usingStemming and self.usingCorrection :
            new_words = []
            
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            
            for word in words:
                if word in self.feature_space:
                    new_words.append(word)
                else:
                    word = self.corrector.correct(word)
                    new_words.append(word)
            return new_words
        
        elif not self.usingNegation and self.usingStemming and not self.usingCorrection:
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            words = [self.stem_wordnet(word) for word in words]
            return words
            
        elif not self.usingNegation and self.usingStemming and self.usingCorrection:
            new_words = []
            
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            words = [self.stem_wordnet(word) for word in words]
            
            for word in words:
                if word in self.feature_space:
                    new_words.append(word)
                else:
                    word = self.corrector.correct(word)
                    new_words.append(word)
            return new_words
        
        elif self.usingNegation and not self.usingStemming  and not self.usingCorrection:
            words = self.negation(" ".join(words))
            
            # Remove punctuations
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            return words

        elif self.usingNegation and not self.usingStemming and self.usingCorrection:
            new_words = []
            words = self.negation(" ".join(words))
            
            # Remove punctuations
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            
            for word in words:
                if word in self.feature_space:
                    new_words.append(word)
                else:
                    if len(word)>=5 and word[-3:]=='neg':
                        word = self.corrector.correct(word[:-3])
                        new_words.append(word + 'neg')
                    else:
                        word = self.corrector.correct(word)
                        new_words.append(word)
            return new_words
            # Deal with negation
        
        elif self.usingNegation and self.usingStemming and not self.usingCorrection:
            new_words = []
            words = self.negation(" ".join(words))
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            
            for word in words:
                if len(word)>=5 and word[-3:]=='neg':
                    stem_word = self.stem_wordnet(word[:-3])+'neg'
                    if stem_word in self.feature_space:
                        new_words.append(stem_word)
                else:
                    stem_word = self.stem_wordnet(word)
                    if stem_word in self.feature_space:
                        new_words.append(stem_word)
            return new_words
            
        else:
            new_words = []
            words = self.negation(" ".join(words))
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            
            for word in words:
                if len(word)>=5 and word[-3:]=='neg':
                    stem_word = self.stem_wordnet(word[:-3])+'neg'
                    if stem_word in self.feature_space:
                        new_words.append(stem_word)
                    else:
                        stem_word = self.corrector.correct(stem_word[:-3])
                        if stem_word+'neg' in self.feature_space:
                            new_words.append(stem_word)
                        
                else:
                    stem_word = self.stem_wordnet(word)
                    if stem_word in self.feature_space:
                        new_words.append(stem_word)
                    else:
                        stem_word = self.corrector.correct(stem_word)
                        if stem_word in self.feature_space:
                            new_words.append(stem_word)
            return new_words
            
    def extract_features_training(self, words):
        if not self.usingNegation and not self.usingStemming:
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            return words
        
        elif not self.usingNegation and self.usingStemming:
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            words = [self.stem_wordnet(word) for word in words]
            return words
        
        elif self.usingNegation and not self.usingStemming:
            words = self.negation(" ".join(words))
            
            # Remove punctuations
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            return words
        
        elif self.usingNegation and self.usingStemming:
            new_words = []
            words = self.negation(" ".join(words))
            words = re.findall(r"[a-zA-Z']+[a-zA-Z0-9]", " ".join(words))
            
            for word in words:
                if len(word)>=5 and word[-3:]=='neg':
                    stem_word = self.stem_wordnet(word[:-3])+'neg'
                    new_words.append(stem_word)
                else:
                    stem_word = self.stem_wordnet(word)
                    new_words.append(stem_word)
            return new_words
            
    def build_feature_vector(self, words):
        feature_vector = [0] * len(self.feature_space)
        for word in words:
            if word in self.feature_space:
                feature_vector[self.feature_space.index(word)] += 1
            #else:
                #print word
        return feature_vector
        
    def build_feature_dict(self, words):
           feature_dict = {}
           for word in self.feature_space:
               feature_dict["count({})".format(word)] = words.count(word)
               #feature_dict["has({})".format(word)] = (word in words)
           return feature_dict

    def build_training_set(self, training_x, training_y):
            training_tup_lst = []
            for i in xrange(len(training_x)):
                training_tup_lst.append((training_x[i], training_y[i]))
            training_set = nltk.classify.apply_features(self.build_feature_dict, training_tup_lst, labeled=True)
            return training_set

