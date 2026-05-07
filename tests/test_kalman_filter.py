import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class KalmanFilter2DTests(unittest.TestCase):
    def test_constant_velocity_motion_converges_to_truth(self):
        from src.kalman_filter import KalmanFilter2D

        kf = KalmanFilter2D(dt=0.5, process_noise=0.05, measurement_noise=0.5)

        # Gerçek hareket: (0,0)'dan başlayıp vx=2 m/s, vy=1 m/s sabit hız
        true_vx, true_vy = 2.0, 1.0
        positions = [(true_vx * 0.5 * i, true_vy * 0.5 * i) for i in range(20)]

        rng = np.random.default_rng(0)
        for x_true, y_true in positions:
            kf.predict()
            noisy = (x_true + rng.normal(0, 0.3), y_true + rng.normal(0, 0.3))
            kf.update(noisy)

        x_est, y_est, vx_est, vy_est = kf.get_state()
        # Son adımda konum gerçek değere yakın olmalı (gürültü payı altında)
        self.assertAlmostEqual(x_est, positions[-1][0], delta=1.0)
        self.assertAlmostEqual(y_est, positions[-1][1], delta=1.0)
        # Hız tahmininin de gerçeğe yakınsadığı doğrulanmalı
        self.assertAlmostEqual(vx_est, true_vx, delta=0.5)
        self.assertAlmostEqual(vy_est, true_vy, delta=0.5)

    def test_higher_measurement_noise_widens_innovation(self):
        from src.kalman_filter import KalmanFilter2D

        kf_low = KalmanFilter2D(dt=0.5, process_noise=0.1, measurement_noise=0.1)
        kf_high = KalmanFilter2D(dt=0.5, process_noise=0.1, measurement_noise=10.0)

        # Aynı uçurum-noktası ölçümü ver
        z = (5.0, 5.0)
        kf_low.predict(); kf_low.update(z)
        kf_high.predict(); kf_high.update(z)

        # Yüksek measurement noise → tahmin daha az kaymalı (ölçüme daha az güveniyor)
        x_low, y_low, _, _ = kf_low.get_state()
        x_high, y_high, _, _ = kf_high.get_state()

        self.assertGreater(abs(x_low) + abs(y_low), abs(x_high) + abs(y_high))

    def test_predict_only_propagates_velocity(self):
        from src.kalman_filter import KalmanFilter2D

        kf = KalmanFilter2D(dt=1.0)
        kf.x = np.array([[0.0], [0.0], [3.0], [4.0]])  # vx=3, vy=4

        kf.predict()
        x, y, vx, vy = kf.get_state()
        self.assertAlmostEqual(x, 3.0)
        self.assertAlmostEqual(y, 4.0)
        self.assertAlmostEqual(vx, 3.0)
        self.assertAlmostEqual(vy, 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
