from collections import Counter
import numpy as np
import re

class TF_IDF:
    def __init__(self, tf:bool, idf:bool) -> None:
        self.use_tf = tf
        self.use_idf = idf
        self.data = None
        self.vectorized_data = None
        self.words = Counter()
        self.terms_dict = dict()
        self.test_data = None
        self.idf_vector

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


    def tf(self):
        temp_words = Counter()
        for i in range(len(self.test_data)):
            temp_words.update(self.test_data[i])
            sum_of_terms = sum([v for k, v in temp_words.items()])
            for (k, v) in temp_words.items():
                if k in self.terms_dict:
                    self.vectorized_data[i][self.terms_dict[k]] = v/sum_of_terms

    def no_tf(self):
        temp_words = Counter()
        for i in range(len(self.test_data)):
            temp_words.update(self.test_data[i])
            for (k, v) in temp_words.items():
                if k in self.terms_dict:
                    self.vectorized_data[i][self.terms_dict[k]] = 1

   ' def idf(self):'


   
    def fit(self, data):
        self.preprocess_data(data=data)
    

    def predict(self, test_data):
        self.test_data = [re.findall(r'\w+', i) for i in test_data]
        self.vectorized_data = np.zeros(shape=(len(self.test_data), len(self.terms_dict)))
        if self.use_tf:
            self.tf()
        else:
            self.no_tf()
        

