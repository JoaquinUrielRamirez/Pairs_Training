import numpy as np

class KalmanFilterReg():
    def __init__(self):
        self.x = np.array([1, 1])  # Initial Observation
        self.A = np.eye(2)  # Transition matrix
        self.Q = np.ones(2) * 10  # covariance matrix in estimations
        self.R = np.array([[1]]) * 100  # error in observations
        self.P = np.eye(2) * 10  # predicted error covariance matrix

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        C = np.array([[1, x]])  # Observation (1, 2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)  # Kalman Gain
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y - C @ self.x)