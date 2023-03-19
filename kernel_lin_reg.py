import numpy as np

class Kernel_Linear_Regression:
    def __init__(self, kernel_type="rbf", kernel_gamma=1.0, kernel_degree=3, 
                 epochs=200, batch_size=10, learning_rate=0.01, l2=0.0) -> None:
        self.betas = None
        self.bias = 0
        self.k_train = []
        self.k_test = []
        self.type = 1 if kernel_type == "rbf" else 0
        self.gamma = kernel_gamma
        self.degree = kernel_degree
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.x = None
        self.rng = np.random.RandomState()

    def init_kernel(self, x:np.array, train=True):
        if train:
            self.betas = np.zeros(self.x.size)
        for i in range(len(x)):
            temp = []
            for j in range(len(self.x)):
                temp.append(Kernel_Linear_Regression.kernel_func(x[i], self.x[j], self.degree, self.gamma, self.type))
            if (train):
                self.k_train.append(temp)
            else:
                self.k_test.append(temp)
            
    def fit(self, x: np.array, y):
        self.x = x
        Kernel_Linear_Regression.init_kernel(self, x)
        for epoch in range(self.epochs):
            permutation = self.rng.permutation(x.shape[0])
            for i in range(int(x.shape[0]/self.batch_size)):
                gradients = []
                for j in range(self.batch_size):
                    temp = permutation[i*self.batch_size+j]
                    t_i = y[permutation[i*self.batch_size+j]]
                    gradient = sum([self.betas[j]*self.k_train[temp][j] for j in range(len(self.betas))])+self.bias - t_i
                    gradients.append(gradient)
                gradient = sum(gradients)/len(gradients)
                for j in range(self.batch_size):
                    temp = permutation[i*self.batch_size+j]
                    self.betas[temp] = self.betas[temp] - (self.learning_rate/self.batch_size)*(gradients[j] + self.l2*self.betas[temp])
                for j in range(len(self.betas)):
                    if j not in permutation[i*self.batch_size:(i+1)*self.batch_size]:
                        self.betas[j] = self.betas[j] - self.learning_rate/self.batch_size*(self.l2*self.betas[j])
                self.bias = self.bias - sum(gradients)/self.batch_size*self.learning_rate

    def predict(self, x_test:np.array):
        self.k_test = []
        Kernel_Linear_Regression.init_kernel(self, x_test, train=False)
        predictions = [sum([self.betas[i]*self.k_test[j][i] for i in range(len(self.betas))])+self.bias 
                       for j in range(len(x_test))]
        return predictions

    @staticmethod
    def kernel_func(x, z, degree, gamma, kernel):
        if kernel:
            return np.exp(-gamma*np.square(np.linalg.norm(x-z)))
        else:
            return pow(gamma * np.matmul(np.transpose(x), z) + 1, degree)