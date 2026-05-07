"""
test_metrics.py — eval/ metric altyapısı birim testleri.

Doğrulama senaryoları:
  1. Mükemmel tracker: MOTA=1.0, IDSW=0
  2. Tümü kaçırılan: MOTA <= 0, FN=GT boyutu
  3. Tümü FP: MOTA <= 0
  4. Boş GT + boş hypothesis: sıfır metrik
  5. Detection: mükemmel tahmin → mAP@0.5 = 1.0
  6. Detection: boş tahmin → mAP@0.5 = 0.0
  7. SyntheticGTGenerator: frame sayısı + radar doğruluğu
  8. Evaluator entegrasyon: full pipeline mükemmel senaryo
"""

import sys
import math
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import compute_mot_metrics, MM_AVAILABLE
from eval.detection_metrics import compute_detection_metrics, COCO_AVAILABLE
from eval.ground_truth import SyntheticGTGenerator
from eval.evaluator import Evaluator


@unittest.skipUnless(MM_AVAILABLE, "motmetrics kurulu değil")
class MOTMetricsTests(unittest.TestCase):

    def _make_frames(self, ids_per_frame, boxes_per_frame):
        return [
            {"frame_id": i, "ids": ids_per_frame[i], "boxes": boxes_per_frame[i]}
            for i in range(len(ids_per_frame))
        ]

    def test_perfect_tracker_mota_one(self):
        box = (10.0, 10.0, 50.0, 50.0)
        gt = self._make_frames([[0], [0], [0]], [[box], [box], [box]])
        hyp = self._make_frames([[0], [0], [0]], [[box], [box], [box]])
        result = compute_mot_metrics(gt, hyp)
        self.assertAlmostEqual(result["MOTA"], 1.0, places=3)
        self.assertEqual(result["IDSW"], 0)
        self.assertEqual(result["FP"], 0)
        self.assertEqual(result["FN"], 0)

    def test_all_missed_negative_mota(self):
        box = (10.0, 10.0, 50.0, 50.0)
        gt = self._make_frames([[0], [0], [0]], [[box], [box], [box]])
        hyp = self._make_frames([[], [], []], [[], [], []])
        result = compute_mot_metrics(gt, hyp)
        self.assertLessEqual(result["MOTA"], 0.0)
        self.assertEqual(result["FN"], 3)
        self.assertEqual(result["FP"], 0)

    def test_all_false_positives(self):
        box = (10.0, 10.0, 50.0, 50.0)
        box2 = (200.0, 200.0, 50.0, 50.0)  # GT ile örtüşmüyor
        gt = self._make_frames([[0], [0]], [[box], [box]])
        hyp = self._make_frames([[99], [99]], [[box2], [box2]])
        result = compute_mot_metrics(gt, hyp)
        self.assertLessEqual(result["MOTA"], 0.0)
        self.assertGreater(result["FP"], 0)
        self.assertGreater(result["FN"], 0)

    def test_empty_gt_and_hyp(self):
        gt = self._make_frames([[], []], [[], []])
        hyp = self._make_frames([[], []], [[], []])
        result = compute_mot_metrics(gt, hyp)
        self.assertIsInstance(result["MOTA"], float)
        self.assertEqual(result["FP"], 0)
        self.assertEqual(result["FN"], 0)

    def test_single_id_switch(self):
        box = (10.0, 10.0, 40.0, 40.0)
        gt = self._make_frames([[0], [0]], [[box], [box]])
        hyp = self._make_frames([[1], [2]], [[box], [box]])  # ID değişiyor
        result = compute_mot_metrics(gt, hyp)
        self.assertGreaterEqual(result["IDSW"], 1)

    def test_two_targets_tracked_correctly(self):
        b1 = (0.0, 0.0, 30.0, 30.0)
        b2 = (200.0, 200.0, 30.0, 30.0)
        gt = self._make_frames([[0, 1], [0, 1]], [[b1, b2], [b1, b2]])
        hyp = self._make_frames([[0, 1], [0, 1]], [[b1, b2], [b1, b2]])
        result = compute_mot_metrics(gt, hyp)
        self.assertAlmostEqual(result["MOTA"], 1.0, places=3)
        self.assertEqual(result["num_objects"], 2)


@unittest.skipUnless(COCO_AVAILABLE, "pycocotools kurulu değil")
class DetectionMetricsTests(unittest.TestCase):

    def _minimal_coco_gt(self):
        return {
            "images": [{"id": 1, "width": 640, "height": 480}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [10.0, 10.0, 100.0, 100.0],
                "area": 10000.0, "iscrowd": 0,
            }],
            "categories": [{"id": 1, "name": "Drone"}],
        }

    def test_perfect_prediction_map_05(self):
        gt = self._minimal_coco_gt()
        preds = [{"image_id": 1, "category_id": 1,
                  "bbox": [10.0, 10.0, 100.0, 100.0], "score": 1.0}]
        result = compute_detection_metrics(gt, preds)
        self.assertAlmostEqual(result["mAP@0.5"], 1.0, places=3)
        self.assertAlmostEqual(result["mAP@0.5:0.95"], 1.0, places=2)

    def test_empty_predictions_zero_map(self):
        gt = self._minimal_coco_gt()
        result = compute_detection_metrics(gt, [])
        self.assertAlmostEqual(result["mAP@0.5"], 0.0, places=3)

    def test_wrong_class_zero_map(self):
        gt = self._minimal_coco_gt()
        preds = [{"image_id": 1, "category_id": 2,  # yanlış sınıf
                  "bbox": [10.0, 10.0, 100.0, 100.0], "score": 1.0}]
        result = compute_detection_metrics(gt, preds)
        self.assertAlmostEqual(result["mAP@0.5"], 0.0, places=3)

    def test_low_iou_prediction_fails_at_05(self):
        gt = self._minimal_coco_gt()
        preds = [{"image_id": 1, "category_id": 1,
                  "bbox": [200.0, 200.0, 100.0, 100.0], "score": 1.0}]
        result = compute_detection_metrics(gt, preds)
        self.assertLess(result["mAP@0.5"], 0.1)


class SyntheticGTTests(unittest.TestCase):

    def test_frame_count_correct(self):
        gen = SyntheticGTGenerator(seed=0)
        frames = gen.generate_sequence(n_frames=30, n_targets=2)
        self.assertEqual(len(frames), 30)

    def test_frame_ids_sequential(self):
        gen = SyntheticGTGenerator(seed=1)
        frames = gen.generate_sequence(n_frames=20, n_targets=1)
        ids = [f.frame_id for f in frames]
        self.assertEqual(ids, list(range(20)))

    def test_radar_detection_range_accuracy(self):
        gen = SyntheticGTGenerator(seed=7, radar_noise_m=1.0)
        frames = gen.generate_sequence(n_frames=50, n_targets=1)
        errors = []
        for f in frames:
            for det in f.radar_detections:
                err_km = abs(det["range_km"] - det["range_km_true"])
                errors.append(err_km)
        if errors:
            mean_err_m = sum(errors) / len(errors) * 1000
            self.assertLess(mean_err_m, 20.0, "Radar mesafe hatası beklenen sınırı aşıyor")

    def test_radar_dropout_works(self):
        gen = SyntheticGTGenerator(seed=3)
        dropout_frames = [0, 1, 2, 3, 4]
        frames = gen.generate_sequence(n_frames=20, n_targets=2,
                                       radar_dropout_frames=dropout_frames)
        for fid in dropout_frames:
            self.assertEqual(frames[fid].radar_detections, [],
                             f"Frame {fid} dropout'ta radar verisi olmamalı")

    def test_to_mot_format_perfect_mota(self):
        if not MM_AVAILABLE:
            self.skipTest("motmetrics yok")
        gen = SyntheticGTGenerator(seed=5, camera_range_m=2000)
        frames = gen.generate_sequence(n_frames=30, n_targets=2)
        gt_mot, hyp_mot = gen.to_mot_format(frames)
        result = compute_mot_metrics(gt_mot, hyp_mot)
        self.assertAlmostEqual(result["MOTA"], 1.0, places=3,
                               msg="Mükemmel tracker MOTA=1.0 vermeli")
        self.assertEqual(result["IDSW"], 0)

    def test_coco_format_annotation_count(self):
        gen = SyntheticGTGenerator(seed=9, camera_range_m=2000)
        frames = gen.generate_sequence(n_frames=10, n_targets=3)
        coco = gen.to_coco_format(frames)
        total_visible = sum(len(f.track_ids) for f in frames)
        self.assertEqual(len(coco["annotations"]), total_visible)
        self.assertEqual(len(coco["images"]), 10)


@unittest.skipUnless(MM_AVAILABLE and COCO_AVAILABLE, "motmetrics veya pycocotools yok")
class EvaluatorIntegrationTests(unittest.TestCase):

    def test_full_pipeline_perfect_scenario(self):
        gen = SyntheticGTGenerator(seed=42, camera_range_m=2000)
        frames = gen.generate_sequence(n_frames=20, n_targets=2)
        gt_mot, hyp_mot = gen.to_mot_format(frames)
        coco_gt = gen.to_coco_format(frames)
        preds = Evaluator.perfect_detections_from_gt(coco_gt)

        ev = Evaluator()
        result = ev.evaluate_full(gt_mot, hyp_mot, coco_gt, preds,
                                  meta={"config": "perfect_baseline"})
        self.assertAlmostEqual(result.mot["MOTA"], 1.0, places=3)
        self.assertAlmostEqual(result.detection["mAP@0.5"], 1.0, places=2)
        self.assertIn("config", result.meta)

    def test_summary_string_non_empty(self):
        gen = SyntheticGTGenerator(seed=0, camera_range_m=2000)
        frames = gen.generate_sequence(n_frames=10, n_targets=1)
        gt_mot, hyp_mot = gen.to_mot_format(frames)
        ev = Evaluator()
        result = ev.evaluate_mot(gt_mot, hyp_mot)
        summary = result.summary()
        self.assertIn("MOTA", summary)
        self.assertGreater(len(summary), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
