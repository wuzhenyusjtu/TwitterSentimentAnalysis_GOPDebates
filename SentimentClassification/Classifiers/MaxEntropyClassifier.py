import nltk.classify
import cPickle

# raw 61.2284622594
# neg 61.2284622594
# stem 61.2284622594
# neg stem 61.2284622594


class MaxEntropyClassifier:
    def __init__(self, needTraining = True):
        self.needTraining = needTraining
        if not self.needTraining:
            try:
                fp = open('maxentropy_trained_model.pickle','rb')
            except (OSError, IOError) as e:
                print "IO / OS error({0}): {1}".format(e.errno, e.strerror)
            self.MEClassifier = cPickle.load(fp)
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
        self.MEClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set,'GIS',trace=3,encoding=None,labels=None,gaussian_prior_sigma=0, max_iter = 3)
        fp = open('maxentropy_trained_model.pickle','wb')
        cPickle.dump(self.MEClassifier, fp)
        fp.close()
    
    def test(self, testing_x):
        print type(testing_x)
        if type(testing_x) is list:
            testing_y = []
            for x in testing_x:
                testing_y.append(self.MEClassifier.classify(x))
            return testing_y
        else:
            return self.MEClassifier.classify(testing_x)
        
        
    def accuracy(self, testing_y, labels):
        print testing_y
        print labels

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