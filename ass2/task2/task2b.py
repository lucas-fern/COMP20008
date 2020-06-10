import math
import pandas
import random
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn import neighbors


def task2b(world='world.csv', life='life.csv'):
    world_data, X_train, X_test, y_train, y_test = generate_splits(world, life)

    # Perform median imputation on missing world data, do train and test data
    # seperately to avoid test data influencing training
    for name in world_data.columns:
        median_impute(X_train, name)
        median_impute(X_test, name)

    # Generate interaction term pairs
    for pair in itertools.combinations(world_data.columns, 2):
        X_train[pair[0] + ' | ' + pair[1]] = (
            X_train[pair[0]] * X_train[pair[1]])
        X_test[pair[0] + ' | ' + pair[1]] = (
            X_test[pair[0]] * X_test[pair[1]])

    # Scale the data including the interaction pair features
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    k_nn_classify(5, X_train, X_test, y_train, y_test,
                  'all features with interaction term pairs')

    RV, _, _ = VAT(X_train)
    x = sns.heatmap(RV, xticklabels=False, yticklabels=False)
    x.set(xlabel='Objects', ylabel='Objects')
    plt.savefig('task2b-VAT.png')

    # Generate feature from k-Means Clustering
    km_train = KMeans(n_clusters=3).fit(X_train[:, 0:20])
    # Add the feature to the array
    X_train = np.c_[X_train, km_train.labels_]

    # Assign test data to the nearest centroid by euclidean distance
    clusters = []
    for feature in X_test:
        feature = feature[0:20]
        closest_centroid = None
        closest_distance = math.inf
        for i in range(len(km_train.cluster_centers_)):
            centroid = km_train.cluster_centers_[i]
            distance = np.linalg.norm(feature-centroid)
            if distance < closest_distance:
                closest_distance = distance
                closest_centroid = i

        clusters.append(closest_centroid)
    X_test = np.c_[X_test, clusters]

    # Extract the first four featuresfrom the original dataset as a sample of
    # the original 20 features
    first_4_train = X_train[:, 0:4]
    first_4_test = X_test[:, 0:4]

    # Perform PCA to extract 4 features
    pca = PCA(n_components=4)
    pca_train = pca.fit_transform(X_train[:, 0:-1])
    pca_test = pca.transform(X_test[:, 0:-1])

    # Extract my choice of 4 features (choices explained in report)
    column_choices = (4, 9, 10, 21)
    my_train = X_train[:, column_choices]
    my_test = X_test[:, column_choices]

    k_nn_classify(5, my_train, my_test, y_train, y_test,
                  'feature engineering')
    k_nn_classify(5, pca_train, pca_test, y_train, y_test,
                  'PCA')
    k_nn_classify(5, first_4_train, first_4_test, y_train, y_test,
                  'first four features')


def generate_splits(world, life):
    """Generate training and testing splits for data from 'world.csv' and
    'life.csv'.
    """
    world_data = pandas.read_csv(world, quotechar='"', skipfooter=5,
                                 na_values='..', index_col='Country Code',
                                 engine='python')
    world_data.drop(['Country Name', 'Time'], axis=1, inplace=True)

    life_data = pandas.read_csv(life, index_col='Country Code')
    life_data.drop(['Country', 'Year'], axis=1, inplace=True)

    # Merge the dataframes using how='inner' so that we dont get any NaN values
    # for life expectancy
    merged_data = pandas.merge(
        world_data, life_data, left_index=True, right_index=True, how='inner')

    class_label = merged_data['Life expectancy at birth (years)']
    features = merged_data.drop('Life expectancy at birth (years)', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        features, class_label, train_size=(2/3), random_state=100)

    return world_data, X_train, X_test, y_train, y_test


def median_impute(array, col_name):
    """Fill NaN values in a column of a DataFrame with the median value
    of that column.
    """
    array[col_name].fillna(array[col_name].median(), inplace=True)


def k_nn_classify(k, X_train, X_test, y_train, y_test, desc):
    """Perform k-NN classification with a specified k and print the accuracy
    to stdout.
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f'Accuracy of {desc}: {accuracy_score(y_test, y_pred):.3f}')


def VAT(R):
    """
    VAT Algorithm taken from Workshop week 8 - modified to comply with PEP8:

    VAT algorithm adapted from matlab version:
    http://www.ece.mtu.edu/~thavens/code/VAT.m

    Args:
        R (n*n double): Dissimilarity data input
        R (n*D double): vector input (R is converted to sq. Euclidean distance)
    Returns:
        RV (n*n double): VAT-reordered dissimilarity data
        C (n int): Connection indexes of MST in [0,n)
        I (n int): Reordered indexes of R, the input data in [0,n)
    """

    R = np.array(R)
    N, M = list(R.shape)
    if N != M:
        R = squareform(pdist(R))

    J = list(range(0, N))

    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)

    i_ = i[j]
    del J[i_]

    y = np.min(R[i_, J], axis=0)
    j = np.argmin(R[i_, J], axis=0)

    i_ = [i_, J[j]]
    J = [e for e in J if e != J[j]]

    C = [1, 1]
    for _ in range(2, N-1):
        y = np.min(R[i_, :][:, J], axis=0)
        i = np.argmin(R[i_, :][:, J], axis=0)
        j = np.argmin(y)
        y = np.min(y)
        i_.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])

    y = np.min(R[i_, :][:, J], axis=0)
    i = np.argmin(R[i_, :][:, J], axis=0)

    i_.extend(J)
    C.extend(i)

    RI = list(range(N))
    for idx, val in enumerate(i_):
        RI[val] = idx

    RV = R[i_, :][:, i_]

    return RV.tolist(), C, i_


task2b(world=r'ass2\task2\world.csv', life=r'ass2\task2\life.csv')
