
import numpy as np

np.random.seed(42)


class BHPSOGWO():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=100):

        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter

        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = np.zeros(self.num_dim)
        self.X_beta = np.zeros(self.num_dim)
        self.X_delta = np.zeros(self.num_dim)

        self.C1 = 0.5
        self.C2 = 0.5
        self.C3 = 0.5

        self.X = (np.random.uniform(
            size=[self.num_particle, self.num_dim]) > 0.5)*1.0

        self.gBest_curve = np.zeros(self.max_iter)
        self.gBest_score = 0

        self.V = 0.3 * np.random.normal(size=[self.num_particle, self.num_dim])

        self.w = 0.5 + np.random.uniform()/2

        self._iter = 0

        self._itter = self._iter + 1

    def opt(self):
        while(self._iter < self.max_iter):
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i, :])

                if score < self.score_alpha:

                    self.score_alpha = score.copy()
                    self.X_alpha = self.X[i, :].copy()

                if score > self.score_alpha and score < self.score_beta:

                    self.score_beta = score.copy()
                    self.X_beta = self.X[i, :].copy()

                if score > self.score_alpha and score > self.score_beta and score < self.score_delta:
                    self.score_delta = score.copy()
                    self.X_delta = self.X[i, :].copy()

            a = 2 - 2*self._iter/self.max_iter

            for i in range(self.num_particle):
                r1 = np.random.uniform(size=[self.num_dim])
                A1 = 2*a*r1 - a
                D_alpha = np.abs(self.C1*self.X_alpha -
                                 self.w*self.X[i, :])

                X1 = self.X_alpha - A1*D_alpha

                r1 = np.random.uniform(size=[self.num_dim])
                A2 = 2*a*r1 - a  
                D_beta = np.abs(self.C2*self.X_beta -
                                self.w*self.X[i, :])
                X2 = self.X_beta - A2*D_beta

                r1 = np.random.uniform(size=[self.num_dim])
                A3 = 2*a*r1 - a  
                D_delta = np.abs(self.C3*self.X_delta -
                                 self.w*self.X[i, :])
                X3 = self.X_delta - A3*D_delta

                
                r1 = np.random.uniform(size=[self.num_dim])
                r2 = np.random.uniform(size=[self.num_dim])
                r3 = np.random.uniform(size=[self.num_dim])
                self.V[i, :] = self.w*(self.V[i, :]
                                       + self.C1*r1*(X1-self.X[i, :])
                                       + self.C2*r2*(X2-self.X[i, :])
                                       + self.C3*r3*(X3-self.X[i, :]))
                
                self.X[i, :] = self.sigmoid((X1+X2+X3)/3) + self.V[i, :]

                self.X[i, :] = (self.X[i, :] >= np.random.uniform(
                    size=[self.num_dim])) * 1.0
                
            self._iter = self._iter + 1
            self.gBest_curve[self._iter-1] = self.score_alpha.copy()
            self.gBest_score = self.score_alpha.copy()

    def return_gbest_curve(self):
        return self.gBest_curve

    def sigmoid(self, x):
        return 1/(1+np.exp(-10*(x-0.5)))
