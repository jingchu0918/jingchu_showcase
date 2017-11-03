# K-means

import random


# the functions returns a tuple containing two lists: the first list
# should be a list of instance (themselves represented as lists) and the
# second list should be a list of labels.
def read_data(filename):
    """ Reads instances and labels from a file. """

    f = open(filename, 'r')
    instances = []
    labels = []

    for line in f:

        # read both feature values and label
        instance_and_label = [float(x) for x in line.split()]
        
        # Remove label (last item) from instance_and_label and append it
        # to labels
        labels.append(instance_and_label[-1])
        instance_and_label.pop()
        
        # Append the instance to instances
        instances.append(instance_and_label)

    return instances, labels



# the function returns the number of unique elements in the list labels
def num_unique_labels(labels):
    """ Return number of unique elements in the list labels. """
    
    return len(set(labels))


# the function choose K centers from the given instances using the
# initialization used by the kmeans++ algorithm as described, for example, in:
# http://en.wikipedia.org/wiki/K-means%2B%2B#Initialization_algorithm
# (Steps 1 through 4)
def kmeans_plus_plus(instances, K):
    """ Choose K centers from instances using the kmeans++ initialization. """
    centers = random.sample(instances, 1)
    # The first center point is chosen uniformly at random among all points
    k = 1
    # k is the number of center points chosen
    n=len(instances)
    
    while k < K :
        distance = []
        for i in range(n):
            d = [euclidean_squared(instances[i],centers[j]) for j in range(k)]
            # Compute the squared euclidean distance between each data point 
            # and the all center points
            distance.append(min(d))
            # Append the minimum distance to the list 'distance'
        u = random.random()
        # Generate a random number from [0,1)
        
        for r in range(n):
            sumdist = sum(distance)
            if r == 0 :
                cum1 = 0.0
                cum2 = distance[0] / sumdist
            else :
                cum1 = sum(distance[0:r]) / sumdist
                cum2 = sum(distance[0:r+1]) / sumdist
            if u >= cum1 and u < cum2:
                break
        # Sample the index r from 0 to n-1 with probabilities proportional to 
        # the corresponding distance

        centers.append(instances[r])
        k = k+1
        # Choose a new center, using a weighted probability distribution where a
        # point is chosen with probability proportional to nearest squared distance
    
    return centers


def euclidean_squared(p1, p2):
    """ Return squared Euclidean distance between two points p1 and p2. """

    return sum([abs(x-y)**2 for (x, y) in zip(p1, p2)])



# the function definition below returns a list cluster_ids such that cluster_idx[i]
# is the index of the center closest to instances[i].
def assign_cluster_ids(instances, centers):
    """ Assigns each instance the id of the center closest to it. """

    n = len(instances)
    cluster_ids = n*[0]  # create list of zeros

    for i in range(n):

        # Compute distances of instances[i] to each of the centers using a list
        # comprehension. Make use of the euclidean_squared function defined
        # above.
        distances = [euclidean_squared(instances[i],centers[j]) for j in range(len(centers))]

        # Find the minimum distance.
        min_distance = min(distances)

        # Set the cluster id to be the index at which min_distance
        # is found in the list distances.
        cluster_ids[i] = distances.index(min_distance)

    return cluster_ids


# the function recomputes the centers given cluster ids of instances.
def recompute_centers(instances, cluster_ids, centers):
    """ Compute centers (means) given cluster ids. """

    K = len(centers)
    n = len(cluster_ids)

    for i in range(K):

        # Find indices of of those instances whose cluster id is i.
        # Use a single list comprehension.
        one_cluster = [j for j in range(n) if cluster_ids[j] == i]
        cluster_size = len(one_cluster)
        if cluster_size == 0:  # empty cluster
            raise Exception("kmeans: empty cluster created.")

        # Suppose one_cluster is [i1, i2, i3, ... ]
        # Compute the mean of the points instances[i1], instances[i2], ...
        # using a call to reduce().
        # Supply the right 1st arg: a lambda function (this should take two
        # points [represented as lists] as arguments and return their sum) and
        # the right 2nd arg: a list (computed using a list comprehension)
        sum_cluster = reduce(lambda x, y: [p+q for p,q in zip(x,y)], [instances[j] for j in one_cluster])
        centers[i] = [x/cluster_size for x in sum_cluster]


def cluster_using_kmeans(instances, K, init='random'):
    """ Cluster instances using the K-means algorithm.

    The init argument controls the initial clustering.
    """

    err_message = 'Expected init to be "random" or "kmeans++", got %s'
    if init != 'random' and init != 'kmeans++':
        raise Exception(err_message % init)

    if init == 'random':
        # Choose initial centers at random from the given instances
        centers = random.sample(instances, K)
    else:
        # Assign clusters using the kmeans++ enhancement.
        centers = kmeans_plus_plus(instances, K)

    # create initial cluster ids
    cluster_ids = assign_cluster_ids(instances, centers)

    converged = False
    while not converged:

        # recompute centers; note function returns None, modifies centers
        # directly
        recompute_centers(instances, cluster_ids, centers)

        # re-assign cluster ids
        new_cluster_ids = assign_cluster_ids(instances, centers)

        if new_cluster_ids == cluster_ids:  # no change in clustering
            converged = True
        else:
            cluster_ids = new_cluster_ids

    return cluster_ids, centers


def main():

    data_file = 'seeds_dataset.txt'
    instances, labels = read_data(data_file)
    print 'Read %d instances and %d labels from file %s.' \
        % (len(instances), len(labels), data_file)

    if len(instances) != len(labels):
        raise Exception('Expected equal number of instances and labels.')
    else:
        n = len(instances)

    # Find number of clusters by finding out how many unique elements are there
    # in labels.
    K = num_unique_labels(labels)
    print 'Found %d unique labels.' % K

    # Run k-means clustering to cluster the instances.
    cluster_ids, centers = cluster_using_kmeans(instances, K)

    # Print the provided labels and the found clustering
    print "Done with kmeans.\nPrinting instance_id, label, cluster_id."
    for i in range(n):
        print '%3d %2d %2d' % (i, labels[i], cluster_ids[i])

    # Now run k-means using kmeans++ initialization
    cluster_ids, centers = cluster_using_kmeans(instances, K, 'kmeans++')

    # Print the provided labels and the found clustering
    print "Done with kmeans++.\nPrinting instance_id, label, cluster_id."
    for i in range(n):
        print '%3d %2d %2d' % (i, labels[i], cluster_ids[i])


if __name__ == '__main__':
    main()
