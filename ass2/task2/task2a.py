import pandas
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


def task_2a(world='world.csv', life='life.csv'):
    """Perform 5-NN, 10-NN and decision tree classification, printing accuracy
    and exporting the mean, median and varance of each feature in the test set
    to 'task1a.csv'.
    """
    world_data, X_train, X_test, y_train, y_test = generate_splits(world, life)

    feature_measures = pandas.DataFrame(
        columns=['feature', 'median', 'mean', 'variance'])
    feature_measures.set_index('feature', inplace=True)

    # Perform median imputation on missing world data, do train and test data
    # seperately to avoid test data influencing training
    for name in world_data.columns:
        feature_measures = feature_measures.append(
            pandas.Series({'median': X_train[name].median()}, name=name))
        median_impute(X_train, name)
        median_impute(X_test, name)

    scaler = preprocessing.StandardScaler().fit(X_train)

    # Record the mean and variance of each variable
    i = 0
    for row in feature_measures.index:
        feature_measures.loc[row]['mean'] = scaler.mean_[i]
        feature_measures.loc[row]['variance'] = scaler.var_[i]
        i += 1

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    decision_tree_classify(X_train, X_test, y_train, y_test)
    k_nn_classify(5, X_train, X_test, y_train, y_test)  # 5-NN classification
    k_nn_classify(10, X_train, X_test, y_train, y_test)  # 10-NN classification

    feature_measures.to_csv('task2a.csv')


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


def decision_tree_classify(X_train, X_test, y_train, y_test):
    """Perform decision tree classification with a maximum depth of 4. Results
    may be inconsistent across trials as the random seed is not set.
    """
    dt = DecisionTreeClassifier(max_depth=4)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    print(f'Accuracy of decision tree: {accuracy_score(y_test, y_pred):.3f}')


def k_nn_classify(k, X_train, X_test, y_train, y_test):
    """Perform k-NN classification with a specified k and print the accuracy
    to stdout.
    """
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f'Accuracy of k-nn (k={k}): {accuracy_score(y_test, y_pred):.3f}')


task_2a(world=r'ass2\task2\world.csv', life=r'ass2\task2\life.csv')
