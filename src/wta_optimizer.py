"""
WTA Optimizer (Weapon-Target Assignment) — Çelik Kubbe İleri Düzey Savunma Modülü
Yöneylem Araştırması (Operations Research) yaklaşımlarıyla, mevcut mühimmatı ve 
batarya kapasitelerini kullanarak tehditleri en optimum şekilde angaje eder.

Optimizasyon Kriterleri:
- Tehdidin XAI Skoru (Yüksek skora öncelik verilir)
- Mesafe ve Vuruş Olasılığı (Hit Probability, P_k)
- Batarya mühimmat durumu

Kullanılan Algoritma: Hungarian Algorithm (Linear Sum Assignment)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger("wta_optimizer")

class BatteryState:
    def __init__(self, battery_id: str, ammo: int, max_range_km: float = 5.0, prob_kill: float = 0.85):
        self.battery_id = battery_id
        self.ammo = ammo
        self.max_range_km = max_range_km
        self.prob_kill = prob_kill

class ThreatState:
    def __init__(self, threat_id: str, range_km: float, threat_score: float):
        self.threat_id = threat_id
        self.range_km = range_km
        self.threat_score = threat_score  # 0-100 arası XAI Skoru

class WTAOptimizer:
    @staticmethod
    def optimize(batteries: list[BatteryState], threats: list[ThreatState]) -> list[tuple[str, str]]:
        """
        Bataryaları tehditlere en optimum şekilde atar.
        Geri dönüş: [(batarya_id, tehdit_id), ...]
        """
        available_batteries = [b for b in batteries if b.ammo > 0]
        if not available_batteries or not threats:
            return []

        # Maliyet Matrisi (Cost Matrix): Satırlar=Bataryalar, Sütunlar=Tehditler
        # Amaç maliyeti (cost) minimize etmektir. Bu yüzden yüksek öncelikli eşleşmelere negatif/düşük maliyet vereceğiz.
        num_b = len(available_batteries)
        num_t = len(threats)
        cost_matrix = np.zeros((num_b, num_t))

        for i, b in enumerate(available_batteries):
            for j, t in enumerate(threats):
                # Vuruş olasılığı mesafeye bağlı düşer
                if t.range_km > b.max_range_km:
                    pk = 0.0  # Menzil dışı
                else:
                    # Basit bir fonksiyon: Menzil sınırına yaklaştıkça P_k lineer düşer
                    pk = b.prob_kill * (1.0 - (t.range_km / (b.max_range_km * 1.5)))

                # Beklenen Değer (Expected Value) = Tehdit Skoru * Vuruş Olasılığı
                expected_value = t.threat_score * pk

                # Scipy linear_sum_assignment minimum maliyeti bulur. 
                # Biz maksimize etmek istediğimiz için değeri eksi (-) ile çarpıyoruz.
                # Menzil dışındaysa çok yüksek bir ceza puanı (penalty) verelim.
                if pk <= 0.01:
                    cost = 10000.0  
                else:
                    cost = -expected_value

                cost_matrix[i, j] = cost

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assignments = []
        for i, j in zip(row_ind, col_ind):
            # Eğer atanan maliyet ceza sınırındaysa (menzil dışıysa), atama yapma
            if cost_matrix[i, j] >= 10000.0:
                continue
            
            bat_id = available_batteries[i].battery_id
            thr_id = threats[j].threat_id
            assignments.append((bat_id, thr_id))
            logger.info(f"WTA Ataması: Batarya {bat_id} -> Hedef {thr_id} (Beklenen Değer: {-cost_matrix[i,j]:.1f})")

        return assignments
