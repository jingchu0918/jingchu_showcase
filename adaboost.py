# Adaboost
#


import math
import numpy as np
from assignment_two_svm \
    import evaluate_classifier, print_evaluation_summary


# the function returns a function, not the
# sign, feature, threshold triple
def weak_learner(instances, labels, dist):

    """ Returns the best 1-d threshold classifier.

    A 1-d threshold classifier is of the form

    lambda x: s*x[j] < threshold

    where s is +1/-1,
          j is a dimension
      and threshold is real number in [-1, 1].

    The best classifier is chosen to minimize the weighted misclassification
    error using weights from the distribution dist.

    """
    
    n, d = instances.shape
    
    def error(j, threshold, s, dist):
        
        """ Calculate the weighted misclassification error for given j and threshold
        in a decision stump. """
        predict = (s * instances[:, j]) < threshold
        
        return sum(dist[predict != labels])
     
    # errormat[k,j,i] is 2*d*n which records the weighted misclssification error 
    # when we use Xj as feature and use the i'th observed value of Xj as 
    # threshold in a decision stump. The third dimension k records the value of 
    # s (k = 0 corresponds to s = 1; k = 1 corresponds to s = -1)
    errormat=np.array([error(j,instances[i,j], s, dist)  for s in [1, -1] for j in range(d) for i in range(n) ])
    errormat.shape = (2, d, n)  
      
    # Find the j, threshold and s which minimized the weighted error
    k, j, i = np.unravel_index(errormat.argmin(),errormat.shape)
    s = -2 * k + 1.
    threshold = instances[i, j]    
    
    return lambda x: s * x[j] < threshold    
        
        

def compute_error(h, instances, labels, dist):

    """ Returns the weighted misclassification error of h.

    Compute weights from the supplied distribution dist.
    
    """
   
    temp = dist[[h(instances[i,]) != labels[i] for i in range(labels.size)]]
    return sum(temp)


# Implement the Adaboost distribution update
# this function returns a probability distribution
def update_dist(h, instances, labels, dist, alpha):

    """ Implements the Adaboost distribution update. """
    
    # sign = 1 if h(Xi) != Yi
    # sign = -1 if h(Xi) == Yi
    sign = np.array([2 * (h(instances[i,]) != labels[i]) - 1 for i in range(labels.size)])
    
    new_dist = dist * np.exp(sign * alpha)
    z = sum(new_dist) # normalization constant
    new_dist = new_dist / z
    
    return new_dist

def run_adaboost(instances, labels, weak_learner, num_iters=20):

    n, d = instances.shape
    n1 = labels.size

    if n1 != n:
        raise Exception('Expected same number of labels as no. of rows in \
                        instances')

    alpha_h = []

    dist = np.ones(n)/n

    for i in range(num_iters):

        print "Iteration: %d" % i
        h = weak_learner(instances, labels, dist)

        error = compute_error(h, instances, labels, dist)

        if error > 0.5:
            break

        alpha = 0.5 * math.log((1-error)/error)

        dist = update_dist(h, instances, labels, dist, alpha)

        alpha_h.append((alpha, h))

    # return a classifier whose output
    # is an alpha weighted linear combination of the weak
    # classifiers in the list alpha_h
    def classifier(point):
        """ Classifies point according to a classifier combination.

        The combination is stored in alpha_h.
        """
        temp = [alpha_h[i][0] * (2 * alpha_h[i][1](point) - 1) for i in range(num_iters)]
        temp = sum(temp)
        if temp > 0:
            return 1
        else:
            return 0
    
    return classifier


def main():
    data_file = 'ionosphere.data'

    data = np.genfromtxt(data_file, delimiter=',', dtype='|S10')
    instances = np.array(data[:, :-1], dtype='float')
    labels = np.array(data[:, -1] == 'g', dtype='int')

    n, d = instances.shape
    nlabels = labels.size

    if n != nlabels:
        raise Exception('Expected same no. of feature vector as no. of labels')

    train_data = instances[:200]  # first 200 examples
    train_labels = labels[:200]  # first 200 labels

    test_data = instances[200:]  # example 201 onwards
    test_labels = labels[200:]  # label 201 onwards

    print 'Running Adaboost...'
    adaboost_classifier = run_adaboost(train_data, train_labels, weak_learner)
    print 'Done with Adaboost!\n'

    confusion_mat = evaluate_classifier(adaboost_classifier, test_data,
                                        test_labels)
    print_evaluation_summary(confusion_mat)

if __name__ == '__main__':
    main()
