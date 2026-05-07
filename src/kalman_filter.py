"""
KalmanFilter — Çelik Kubbe İleri Düzey Hedef Takip (State Estimation) Modülü
Doktora seviyesi hedef takip için 2D Lineer Kalman Filtresi uyarlaması.

Durum Vektörü (State Vector) X: [x, y, vx, vy]^T
Ölçüm Vektörü (Measurement) Z: [x, y]^T
"""

import numpy as np

class KalmanFilter2D:
    def __init__(self, dt=0.5, process_noise=0.1, measurement_noise=1.0):
        self.dt = dt
        
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # Covariance Matrix P (Uncertainty)
        self.P = np.eye(4) * 1000.0
        
        # State Transition Matrix F (Kinematic model)
        self.F = np.array([
            [1, 0, dt, 0 ],
            [0, 1, 0,  dt],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ]
        ])
        
        # Measurement Matrix H (We only measure x and y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process Noise Covariance Q (Uncertainty in the model)
        q = process_noise
        self.Q = np.array([
            [q*(dt**4)/4, 0,           q*(dt**3)/2, 0          ],
            [0,           q*(dt**4)/4, 0,           q*(dt**3)/2],
            [q*(dt**3)/2, 0,           q*(dt**2),   0          ],
            [0,           q*(dt**3)/2, 0,           q*(dt**2)  ]
        ])
        
        # Measurement Noise Covariance R (Uncertainty in the sensor)
        r = measurement_noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ])

    def predict(self):
        """1. Aşama: Bir sonraki durumu tahmin et (Prediction)"""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """2. Aşama: Gelen sensör ölçümüyle durumu güncelle (Correction)"""
        Z = np.array([[z[0]], [z[1]]])
        
        y = Z - np.dot(self.H, self.x) # Innovation (Measurement Residual)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R # Innovation Covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x

    def get_state(self):
        """Mevcut x, y konumunu ve vx, vy hızlarını döndürür."""
        return self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
