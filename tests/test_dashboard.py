"""Integration checks of inference and Streamlit interactions."""
from pathlib import Path
import sys
import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import Predictor

class DashboardTests(unittest.TestCase):
    def test_inference_and_validation(self):
        google = Predictor()
        result = google.predict({"cycles_per_instruction": 1.2, "memory_accesses_per_instruction": .008})
        self.assertEqual({r["Model"] for r in result}, {"Random Forest", "One-Class SVM"})
        engine = Predictor("smd")
        sample = pd.read_csv(ROOT / "examples/hardware-sequence.csv")
        self.assertEqual(len(engine.predict_sequence(sample)), 3)
        with self.assertRaises(ValueError):
            engine.predict_sequence(sample.head(10))
        with self.assertRaises(ValueError):
            engine.predict_sequence(sample.drop(columns="metric_01"))

    def test_page_persistence_and_manual_evaluation(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        app.slider(key="_priority").set_value(220).run()
        app.radio[0].set_value("Hardware").run()
        app.slider(key="_cycles_per_instruction").set_value(2.2).run()
        app.radio[0].set_value("Scheduling").run()
        self.assertEqual(app.slider(key="_priority").value, 220)
        app.radio[0].set_value("Hardware").run()
        self.assertEqual(app.slider(key="_cycles_per_instruction").value, 2.2)
        next(b for b in app.button if b.label == "Evaluate manual hardware →").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["evaluation"]), 2)
        next(b for b in app.button if b.label == "Try bundled hardware example").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["evaluation"]), 3)

if __name__ == "__main__":
    unittest.main()
