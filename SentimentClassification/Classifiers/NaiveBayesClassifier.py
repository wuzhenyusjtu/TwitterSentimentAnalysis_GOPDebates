import nltk.classify
import cPickle

# Training accuracy NB on raw: 80.1239997116
# Training accuracy NB on neg_stem: 79.6770240069
# Training accuracy NB on neg: 80.7800446976
# Training accuracy NB on stem:78.8551654531


class NaiveBayesClassifier:
    def __init__(self, needTraining = True):
        self.needTraining = needTraining
        if not self.needTraining:
            try:
                fp = open('naivebayes_trained_model.pickle','rb')
            except (OSError, IOError) as e:
                print "IO / OS error({0}): {1}".format(e.errno, e.strerror)
            self.NBClassifier = cPickle.load(fp)
            fp.close()
    #Use the ``LazyMap`` class to construct a lazy list-like
    #object that is analogous to ``map(feature_func, toks)``.  In
    #particular, if ``labeled=False``, then the returned list-like
    #object's values are equal to::
    #[feature_func(tok) for tok in toks]

    #If ``labeled=True``, then the returned list-like object's values
    #are equal to::[(feature_func(tok), label) for (tok, label) in toks]
    def train(self, training_x, training_y):
        training_set = []
        for i in xrange(len(training_x)):
            training_set.append((training_x[i], training_y[i]))
        # param labeled_featuresets: A list of classified featuresets,
        # i.e., a list of tuples ``(featureset, label)``.
        self.NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
        fp = open('naivebayes_trained_model.pickle','wb')
        cPickle.dump(self.NBClassifier, fp)
        fp.close()
    
    def test(self, testing_x):
        if type(testing_x) is list:
            testing_y = []
            for x in testing_x:
                testing_y.append(self.NBClassifier.classify(x))
            return testing_y
        else:
            return self.NBClassifier.classify(testing_x)
    def accuracy(self, testing_y, labels):
        if len(testing_y) != len(labels):
            raise Exception('Error! Testing labels and actual lables length not match!')
        correct = 0
        wrong = 0
        self.accuracy = 0.0
        for i in xrange(len(testing_y)):
            if labels[i] == testing_y[i]:
                correct += 1
            else:
                wrong += 1
        self.accuracy = (float(correct) / len(testing_y))*100
        return self.accuracy
    def get_most_inform_features(self, num):
        self.NBClassifier.show_most_informative_features(num)
