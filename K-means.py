import numpy as np
import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, points, k = 1) -> None:
        self.data = points
        self._k_clusters = k
        self.clusterDict = defaultdict(list)
        self.meanDict = defaultdict(list)
        self.get_random_k_centers()

    def get_distance(self, x, y):
        '''
        :point x: numpy array
        :point y: numpy array
        :rtype: float value
        '''
        dis = math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)
        return dis
    
    def get_random_k_centers(self):
        for i in range(self._k_clusters):
            self.meanDict[i].append(self.data[random.randint(0, len(self.data) - 1)])
        return 

    def generate_clusters(self):
        for x in self.data:
            min_dis = float('inf')
            flag = -1
            for i, v in self.meanDict.items():
                cur_dis = self.get_distance(x, v[-1])
                if cur_dis < min_dis:
                    min_dis = cur_dis
                    flag = i
            self.clusterDict[flag].append(x)
        return
    
    def get_means(self):
        '''
        dic: points list of Kth clusters
        '''
        for i, v in self.clusterDict.items():
            self.meanDict[i].append(np.mean(v, axis=0))
        return 
    
    def getDiffCenter(self):
        errors = []
        for i, v in self.meanDict.items():
            errors.append(self.get_distance(v[-1], v[-2]))
        return np.mean(errors)
    
    def visualize(self):
        X = [i[0] for i in self.data]
        Y = [i[1] for i in self.data]
        X_center = [self.meanDict[i][-1][0] for i in range(self._k_clusters)]
        Y_center = [self.meanDict[i][-1][1] for i in range(self._k_clusters)]
        plt.scatter(X, Y)
        plt.scatter(X_center, Y_center, color='red')
        

    
    def run(self, err = 0.1):
        while True:
            self.generate_clusters()
            self.get_means()
            if self.getDiffCenter() < err:
                print('Train finished.')
                break
        self.visualize()
        return

if __name__ == '__main__':
    points = list(np.random.uniform(0,2,(10,2))) + list(np.random.uniform(3,4,(10,2))) + list(np.random.uniform(6,8,(10,2)))
    X = [i[0] for i in points]
    Y = [i[1] for i in points]
    kmeans = Kmeans(points, 3)
    kmeans.run()