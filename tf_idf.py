from collections import Counter
import numpy as np
import re

class TF_IDF:
    def __init__(self, tf:bool, idf:bool) -> None:
        self.use_tf = tf
        self.use_idf = idf
        self.data = None
        self.vectorized_data = None
        self.vectorized_train_data = None
        self.words = Counter()
        self.terms_dict = dict()
        self.test_data = None
        self.idf_vector = None

    def preprocess_data(self, data):
        data = [re.findall(r'\w+', i) for i in data]
        for doc in data:
            self.words.update(doc)
        i = 0
        for k, v in self.words.items():
            if (v > 1):
                self.terms_dict[k] = i
                i += 1
        self.words.clear()
        self.data = data


    def tf(self, data, fit=True):
        temp_words = Counter()
        for i in range(len(data)):
            temp_words.update(data[i])
            sum_of_terms = sum([v for k, v in temp_words.items()])
            for (k, v) in temp_words.items():
                if k in self.terms_dict:
                    if fit:
                        self.vectorized_train_data[i][self.terms_dict[k]] = v/sum_of_terms
                    else:
                        self.vectorized_data[i][self.terms_dict[k]] = v/sum_of_terms

    def no_tf(self, data, fit=True):
        temp_words = Counter()
        for i in range(len(data)):
            temp_words.update(data[i])
            for (k, v) in temp_words.items():
                if k in self.terms_dict:
                    if fit:
                        self.vectorized_train_data[i][self.terms_dict[k]] = 1
                    else:
                        self.vectorized_data[i][self.terms_dict[k]] = 1

    def idf(self, fit=True):
        if fit:
            self.idf_vector = np.sum((self.vectorized_train_data > 0).astype(int), axis=0)
            self.idf_vector = np.log(len(self.data)/(self.idf_vector+1))
            self.vectorized_train_data = np.multiply(self.vectorized_train_data, self.idf_vector)
        else:
            self.vectorized_data = np.multiply(self.vectorized_data, self.idf_vector)

   
    def fit(self, data):
        self.preprocess_data(data=data)
        self.vectorized_data = np.zeros(shape=(len(self.data), len(self.terms_dict)))
        if self.use_tf:
            self.tf(self.data, fit=True)
        else:
            self.no_tf(self.data, fit=True)
        if self.use_idf:
            self.idf(fit=True)
        return self.vectorized_train_data


    def predict(self, test_data):
        self.test_data = [re.findall(r'\w+', i) for i in test_data]
        self.vectorized_data = np.zeros(shape=(len(self.test_data), len(self.terms_dict)))
        if self.use_tf:
            self.tf(self.test_data, fit=False)
        else:
            self.no_tf(self.test_data, fit=False)
        if self.use_idf:
            self.idf(fit=False)
        return self.vectorized_data
        

