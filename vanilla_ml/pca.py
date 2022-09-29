import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA():

    def __init__(self, S):
        self.S = S.T
    
    def decomposition(self):
        self.A, self.D = np.linalg.eig(self.S)

        return self.A, self.D
    
    def explained_variance(self):
        self.cumulative_variance = [i / sum(self.A) for i in self.A]

        return self.cumulative_variance
    
    def screeplot(self):
        plt.plot(self.cumulative_variance)
        plt.xlabel('Components')
        plt.ylabel('Explained Cumulative Variance')
        plt.title('PCA')


if __name__ == '__main__':
    returns = pd.read_csv('/Users/gabestechschulte/Documents/git-repos/University/machine-learning-II/lecture-1/stock_returns.csv')
    S = np.cov(returns.iloc[:, 1:14].values)
    pca = PCA(S)
    eigenvalues, eigenvectors = pca.decomposition()
    variance = pca.explained_variance()
    print(variance)
    pca.screeplot()

