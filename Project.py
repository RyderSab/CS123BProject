"""
Project: Fine Needle Aspirates and Machine Learning for Tumor Diagnosis
Author: Ryder Sabale
Date: May 1, 2025
Version: 1.0

Dependencies:
- Python >= 3.8
- NumPy 2.2.5
- Pandas 2.2.3
- scikit-learn 1.6.1 (for train_test split and comparing our implementation to established implementation)
- Joblib 1.4.2 (For paralleliztion)
"""

# import packages
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from ucimlrepo import fetch_ucirepo


class TreeNode:
    """
    Represents a single node in a decision tree

    Attributes:
        feature (str): The feature used for splitting at this node.
        split_point (float): The value of the feature to split on.
        leaf (int/None): if the node is a leaf, the class label; None otherwise.
        left, right (TreeNode): The left and right child node of this node.
    """
    def __init__(self, feature=None, split_point=None, left=None, right=None, leaf=None):
        self.feature = feature
        self.split_point = split_point
        self.leaf = leaf
        self.left = left
        self.right = right


class myDecisionTreeClassifier:
    """
    My implementation of a decision tree classifier algorithm, supports Gini Impurity and Entropy (Information Gain) as criteria for splitting the tree
    Currently only capable of binary classification, to be worked on in future versions

    Attributes:
        criterion (str): The splitting criterion used to build the tree ('gini' or 'entropy').
        max_depth (int/None): The maximum depth of the tree; None for no limit.
        root (TreeNode): The root of the decision tree.
    """
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def gini_impurity(targets):
        """
        Computes the Gini Impurity for a given set of targets.

        Parameters:
            targets (DataFrame): The target labels.

        Returns:
            float: The Gini Impurity value.
        """
        if (len(targets) == 0):
            return 0

        targets = np.array(targets, dtype=int).flatten()

        p = np.bincount(targets) / len(targets)
        return 1 - np.sum(p ** 2)

    @staticmethod
    def entropy(targets):
        """
        Computes the Entropy for a given set of targets.

        Parameters:
            targets (DataFrame): The target labels.

        Returns:
            float: the Entropy value.
        """
        if (len(targets) == 0):
            return 0

        targets = np.array(targets, dtype=int).flatten()

        p = np.bincount(targets) / len(targets)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def weighted_average_gini_impurity(targets, split_index):
        n = len(targets)
        left_impurity = myDecisionTreeClassifier.gini_impurity(targets[:split_index])
        right_impurity = myDecisionTreeClassifier.gini_impurity(targets[split_index:])
        return (split_index * left_impurity + (n - split_index) * right_impurity) / n

    def information_gain(self, targets, split_index):
        n = len(targets)
        left_entropy = self.entropy(targets[:split_index])
        right_entropy = self.entropy(targets[split_index:])
        weighted_average_entropy = (split_index * left_entropy + (n - split_index) * right_entropy) / n
        return -(self.entropy(targets) - weighted_average_entropy)

    def calculate_split_impurities(self, values, targets):
        split_data = []

        for feature in values.columns:
            sorted_indices = np.argsort(values[feature])
            sorted_feature = values[feature].iloc[sorted_indices]
            sorted_targets = targets.iloc[sorted_indices]

            unique_values = sorted_feature.unique()
            split_points = (unique_values[1:] + unique_values[:-1]) / 2

            for split in split_points:
                split_index = np.searchsorted(sorted_feature, split, side='right')

                if (self.criterion == 'gini'):
                    criterion = myDecisionTreeClassifier.weighted_average_gini_impurity(sorted_targets, split_index)
                else:
                    criterion = myDecisionTreeClassifier.information_gain(sorted_targets, split_index)

                split_data.append({
                    'feature': feature,
                    'split_point': split,
                    'criterion': criterion
                })

        return pd.DataFrame(split_data)

    def build_tree(self, values, targets, depth):
        """
        Recursively builds the decision tree.

        Args:
            values (DataFrame): The feature values.
            targets (DataFrame): The target labels.
            depth (int): The current depth of the tree.

        Returns:
            TreeNode: The root of the subtree.
        """
        if ((len(np.unique(targets)) == 1) or
                (len(values) == 0) or
                (self.max_depth is not None and depth >= self.max_depth)):
            leaf_value = targets.mode().iloc[0] if len(values) > 0 else None
            return TreeNode(leaf=leaf_value)

        split_data = self.calculate_split_impurities(values, targets)
        if split_data.empty:
            leaf_value = targets.mode().iloc[0]
            return TreeNode(leaf=leaf_value)

        bestFeatureSplit = split_data.sort_values('criterion').iloc[0]
        feature = bestFeatureSplit.loc['feature']
        split_point = bestFeatureSplit.loc['split_point']

        left_values = values[values[feature] < split_point]
        right_values = values[values[feature] >= split_point]

        if ((len(left_values) == 0) or (len(right_values) == 0)):
            leaf_value = targets.mode().iloc[0]
            return TreeNode(leaf=leaf_value)

        left_targets = targets.loc[left_values.index]
        right_targets = targets.loc[right_values.index]
        left_values = left_values
        right_values = right_values

        left_child = self.build_tree(left_values, left_targets, depth + 1)
        right_child = self.build_tree(right_values, right_targets, depth + 1)

        return TreeNode(feature, split_point, left_child, right_child)

    def fit(self, values, targets):
        self.root = self.build_tree(values, targets, depth=0)

    def predictRecurse(self, node, value):
        """
        Recursively traverses the decision tree according to feature-splits per node

        Args:
            node (TreeNode): the root of the decision tree
            value (DataFrame): The feature values.

        Returns:
            int: class label of the sample
        """
        if (node.leaf is not None):
            return node.leaf

        if (value[node.feature] <= node.split_point):
            return self.predictRecurse(node.left, value)
        else:
            return self.predictRecurse(node.right, value)

    def predict(self, values):
        return np.array([self.predictRecurse(self.root, row) for _, row in values.iterrows()])


class myRandomForestClassifier:
    """
    My implementation of a random forest classifier using the above custom decision tree.

    Attributes:
        criterion (str): The splitting criterion ('gini' or 'entropy').
        max_features (str): How many features should be used for training each tree ('sqrt' or 'log2'; Uses all features otherwise).
        n_estimators (int): The number of trees in the forest.
        max_depth (int/None): The maximum depth of each tree; None for no limit.
        forest (list): List of trained decision trees.
    """

    def __init__(self, criterion='gini', max_features='sqrt', n_estimators=100, max_depth=None):
        self.criterion = criterion
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.forest = []

    def fit(self, values, targets):
        self.forest = []

        def train_tree(index):
            """
            Trains the random forest by fitting multiple decision trees to subsets of the data.

            Args:
                index (int): The current number of trees in the forest, unused in function

            Returns:
                TreeNode: trained decision tree
            """

            if self.max_features == 'sqrt':
                n_features = int(np.sqrt(values.shape[1]))
            elif self.max_features == 'log2':
                n_features = int(np.log2(values.shape[1]))
            else:
                n_features = values.shape[1]

            feature_subset = values.sample(n=n_features, axis=1, replace=False)
            values_subset = values.sample(n=len(values), replace=True)

            # .reset_index() used due to memory usage problems
            targets_subset = targets.loc[values_subset.index].reset_index(drop=True)
            values_subset = values_subset.reset_index(drop=True)

            tree = myDecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
            tree.fit(values_subset, targets_subset)
            return tree

        self.forest = Parallel(n_jobs=4)(delayed(train_tree)(i) for i in range(self.n_estimators)) #n_job set 4 instead of -1 due to memory usage problems

    def predict(self, values):
        """
        Predicts the class labels for a given set of feature values by recursively traversing each tree and taking the majority vote for each sample's class label.

        Args:
           values (DataFrame): The feature values.

        Returns:
           ndarray: Predicted class labels.
        """

        all_preds = np.array([tree.predict(values) for tree in self.forest])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)


# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

# data preprocessing
bool_map = {"M": 1, "B": 0}
y = y.map(lambda x: bool_map.get(x))

# 0.7 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# model training
rf = myRandomForestClassifier()
rf.fit(X_train, y_train)

# model testing
preds = rf.predict(X_test)

matches = 0
for x in range(len(y_test)):
    if (preds[x] == y_test.iloc[x].item()):
        matches += 1

print(matches / len(y_test))
