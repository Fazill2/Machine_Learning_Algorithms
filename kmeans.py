import numpy as np


class K_means:
    def __init__(self, iterations = 50, n_clusters = 2,kpp=True) -> None:
        self.generator = np.random.RandomState()
        self.data = None
        self.centers = None
        self.clusters = None
        self.iterations = iterations
        self.num_clusters = n_clusters
        self.k_plus_plus = kpp
        self.z = None

    def initialize_centers(self):
        if self.k_plus_plus:
            index = self.generator.randint(len(self.data))
            used_indices = [index]
            self.centers = np.zeros(shape=(self.num_clusters, len(self.data[0])))
            self.centers[0] = self.data[index]
            for i in range(1, self.num_clusters):
                unused_indices = [j for j in range(len(self.data)) if j not in used_indices]
                square_distances = [np.square(np.min([np.linalg.norm(self.data[j]-self.centers[k]) for k in range(i)])) for j in unused_indices]
                used_indices.append(self.generator.choice(unused_indices, p=square_distances/np.sum(square_distances)))
                self.centers[i] = self.data[used_indices[-1]]
        else:
            self.centers = self.data[self.generator.choice(len(self.data), size=self.num_clusters, replace=False)]
        
    def fit(self, data: np.array):
        self.data = data
        self.z = np.zeros(shape=(len(self.data), self.num_clusters))
        self.initialize_centers()
        for _ in range(self.iterations):
            self.clusters = []
            for i in range(len(self.data)):
                for k in range(self.num_clusters):
                    self.z[i][k] = 1 if k == np.argmin([np.linalg.norm(data[i] - self.centers[j]) 
                                                        for j in range(len(self.centers))]) else 0
            for k in range(self.num_clusters):
                self.centers[k] = sum([self.z[i][k]*data[i] for i in 
                                       range(len(self.data))])/sum([self.z[i][k] for i in range(len(self.data))])
            for i in range(len(self.data)):
                self.clusters.append(np.argmax(self.z[i]))

    def return_clusters(self):
        return self.clusters
    def return_centers(self):
        return self.centers
