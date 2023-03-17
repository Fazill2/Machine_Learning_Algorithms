from collections import Counter
from math import log
import numpy as np


class DecisionTree:
    class Node:
        def __init__(self, x: np.array, y, criterion, min_to_split = None, max_leaves = None, max_depth = None, depth = 0):
            self.y = y      # target
            self.x = x      # data
            self.depth = depth
            self.counts = Counter(y)
            self.left = None    # left child
            self.right = None   # right child
            self.max_depth = max_depth  
            self.max_leaves = max_leaves
            self.min_to_split = min_to_split
            self.best_feature = None    # feature along which is the best split ie the one decreasing the criterion the most
            self.best_value = None  # best value of the split
            self.predicted_class = max(self.counts, key=self.counts.get)    # class appearing most commonly in sample
            self.criterion = criterion
            self.criterion_val = DecisionTree.Node.gini_impurity(y) if criterion == "gini" else DecisionTree.Node.entropy(y)  # value of the criterion for this instance
            self.split_gain = None  # decrease of the criterion (useful is max_leaves not none)
            self.right_x = None     # data going to the right node while splitting
            self.right_y = None     # target going to the right node while splitting
            self.left_x = None      # data going to the left node while splitting
            self.left_y = None      # target going to the left node while splitting
            self.find_best_split()

        def find_best_split(self):
            best_feature = None
            best_value = None
            best_gain = None
            for i in range(len(self.x[0])):
                values = np.unique(np.transpose(self.x)[i])
                for value in values:
                    y_right = []
                    x_right = []
                    y_left = []
                    x_left = []
                    for j in range(len(self.x)):
                        if self.x[j][i] < value:
                            y_left.append(self.y[j])
                            x_left.append(self.x[j])
                        else:
                            y_right.append(self.y[j])
                            x_right.append(self.x[j])
                    c_left = DecisionTree.Node.gini_impurity(y_left) if self.criterion == "gini" else DecisionTree.Node.entropy(y_left)
                    c_right = DecisionTree.Node.gini_impurity(y_right) if self.criterion == "gini" else DecisionTree.Node.entropy(y_right)
                    criterion_gain = c_left + c_right
                    if best_gain == None or criterion_gain < best_gain:
                        best_gain = criterion_gain
                        best_feature = i
                        best_value = value
                        self.left_x = x_left
                        self.left_y = y_left
                        self.right_x = x_right
                        self.right_y = y_right
            self.split_gain = best_gain - self.criterion_val
            self.best_feature = best_feature
            self.best_value = best_value
            
        @staticmethod
        def gini_impurity(y):
            y_count = Counter(y)
            n = sum([k for (v, k) in y_count.items()])
            gini = n*sum([(k/n) * (1-k/n) for (v, k) in y_count.items()])
            return gini
        @staticmethod
        def entropy(y):
            y_count = Counter(y)
            n = sum([k for (v, k) in y_count.items()])
            entr = -n*sum([(k/n)*log(k/n)  for (v, k) in y_count.items() if k !=0])
            return entr
    def __init__(self, node: Node, max_leaves=None):
        self.root = node
        self.max_leaves = max_leaves
        self.curr_leaves = 1
        self.best_gini = None
    def train(self):
        if self.max_leaves == None:
            DecisionTree.recursive_split(self, self.root)
        else:
            while self.curr_leaves < self.max_leaves:
                self.best_gini = None
                DecisionTree.traverse_and_find_min(self, self.root)
                DecisionTree.traverse_and_split(self, self.root)
                self.curr_leaves += 1

    def traverse_and_find_min(self, node : Node):
        if node.left is not None:
            DecisionTree.traverse_and_find_min(self, node.left)
            DecisionTree.traverse_and_find_min(self, node.right)
        else:
            if self.best_gini is None or self.best_gini > node.split_gain:
                if (node.max_depth is None or node.depth < node.max_depth) and len(node.x)>node.min_to_split and node.criterion_val != 0:
                    self.best_gini = node.split_gain

    def traverse_and_split(self, node: Node):
        if node.left is not None:
            DecisionTree.traverse_and_split(self, node.left)
            DecisionTree.traverse_and_split(self, node.right)
        else:
            if self.best_gini is not None and  self.best_gini == node.split_gain:
                node.left = DecisionTree.Node(node.left_x, node.left_y, node.criterion, node.min_to_split, node.max_leaves, node.max_depth, node.depth+1)
                node.right = DecisionTree.Node(node.right_x, node.right_y, node.criterion, node.min_to_split, node.max_leaves, node.max_depth, node.depth+1)

    def recursive_split(self, node: Node):
        if (node.max_depth is None or node.depth < node.max_depth) and len(node.x)>node.min_to_split and node.criterion_val != 0:
            node.left = DecisionTree.Node(node.left_x, node.left_y, node.criterion, node.min_to_split, node.max_leaves, node.max_depth, node.depth+1)
            node.right = DecisionTree.Node(node.right_x, node.right_y, node.criterion, node.min_to_split, node.max_leaves, node.max_depth, node.depth+1)
            DecisionTree.recursive_split(self, node.left)
            DecisionTree.recursive_split(self, node.right)
        return

    def predict(self,test_data) -> list:
        predictions = []
        for instance in test_data:
            predictions.append(DecisionTree.true_predict(self, instance, self.root))
        return predictions

    def true_predict(self, instance, node: Node) -> int:
        if node.left is not None:
            if instance[node.best_feature] < node.best_value:
                return DecisionTree.true_predict(self, instance, node.left)
            else:
                return DecisionTree.true_predict(self, instance, node.right)
        else:
            return node.predicted_class
