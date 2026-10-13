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

    def test_automatic_updates_and_persistence(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        app.radio(key="navigation").set_value("Scheduling").run()
        app.slider(key="schedule_priority").set_value(220).run()
        app.radio(key="navigation").set_value("Hardware").run()
        self.assertFalse(app.exception)
        first_score = app.session_state["evaluation"][0]["Score"]
        self.assertEqual(len(app.session_state["history"]), 1)
        app.slider(key="input_cycles_per_instruction").set_value(4.0).run()
        self.assertFalse(app.exception)
        second_score = app.session_state["evaluation"][0]["Score"]
        self.assertNotEqual(first_score, second_score)
        self.assertEqual(len(app.session_state["history"]), 2)
        app.run()
        self.assertEqual(len(app.session_state["history"]), 2)
        app.radio(key="navigation").set_value("Scheduling").run()
        self.assertEqual(app.slider(key="schedule_priority").value, 220)
        app.radio(key="navigation").set_value("Hardware").run()
        self.assertEqual(app.slider(key="input_cycles_per_instruction").value, 4.0)
        app.number_input(key="input_average_cpu").set_value(.8).run()
        self.assertTrue(any("consistent readings" in e.value for e in app.error))
        self.assertEqual(len(app.dataframe), 0)
        app.number_input(key="input_maximum_cpu").set_value(.9).run()
        self.assertFalse(app.error)
        self.assertFalse(app.exception)
        app.radio(key="reading_source").set_value("Hardware sequence").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["evaluation"]), 3)
        fingerprint = app.session_state["last_fingerprint"]
        app.slider(key="sequence_endpoint").set_value(40).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(fingerprint, app.session_state["last_fingerprint"])
        app.selectbox[0].set_value("LSTM Autoencoder").run()
        self.assertFalse(app.exception)

if __name__ == "__main__":
    unittest.main()
