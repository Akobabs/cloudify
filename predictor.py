"""Frozen parent experiment models: seed 42, research_improved run."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).parent
NAMES = {'random_forest': 'Random Forest', 'lstm_autoencoder': 'LSTM Autoencoder', 'one_class_svm': 'One-Class SVM'}

class Predictor:
    def __init__(self, dataset='google'):
        self.dataset = dataset
        directory = ROOT / 'models' / dataset
        self.preprocessor = joblib.load(directory / 'preprocessor.joblib')
        self.models = {kind: joblib.load(directory / f'{kind}.joblib') for kind in ('random_forest', 'one_class_svm')}
        self.metrics = {kind: json.loads((directory / f'{kind}.json').read_text()) for kind in self.models}
        self.lstm = None
        if dataset == 'smd':
            self.metrics['lstm_autoencoder'] = json.loads((directory / 'lstm_autoencoder.json').read_text())

    def transform(self, frame):
        columns = self.preprocessor['raw_features']
        values = self.preprocessor['imputer'].transform(frame.reindex(columns=columns))
        if self.dataset == 'smd':
            df = pd.DataFrame(values)
            window = self.preprocessor['rolling_window']
            values = np.concatenate([values, df.rolling(window, min_periods=1).mean().to_numpy(), df.diff().fillna(0).to_numpy()], axis=1)
        return self.preprocessor['scaler'].transform(values).astype(np.float32)

    def predict(self, values):
        return self.score_tabular(self.transform(pd.DataFrame([values])))

    def score_tabular(self, x):
        results = []
        for kind, model in self.models.items():
            score = float(model.score(x)[-1])
            results.append({'Model': NAMES[kind], 'Outcome': 'FLAGGED' if score > model.threshold else 'NOT FLAGGED', 'Score': score, 'Threshold': float(model.threshold), 'Failure probability': score if kind == 'random_forest' and self.dataset == 'google' else None})
        return results

    def validate_sequence(self, frame):
        columns = self.preprocessor['raw_features']
        if not set(columns).issubset(frame.columns):
            raise ValueError('CSV must contain metric_01 through metric_38, matching the supplied template.')
        if not 30 <= len(frame) <= 1000:
            raise ValueError('Upload 30–1000 chronological observations from one machine.')
        numeric = frame[columns].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError('Hardware observations must be finite numeric values with no missing entries.')
        return numeric

    def predict_sequence(self, frame):
        numeric = self.validate_sequence(frame)
        x = self.transform(numeric)
        results = self.score_tabular(x[-1:])
        if self.lstm is None:
            import tensorflow as tf
            self.lstm = tf.keras.models.load_model(ROOT / 'models/smd/lstm_autoencoder.keras', compile=False)
        window = x[-30:][None, :, :]
        reconstructed = self.lstm(window, training=False).numpy()
        score = float(np.mean(np.square(window[:, -1, :] - reconstructed[:, -1, :])))
        threshold = self.metrics['lstm_autoencoder']['threshold']
        results.insert(1, {'Model': 'LSTM Autoencoder', 'Outcome': 'FLAGGED' if score > threshold else 'NOT FLAGGED', 'Score': score, 'Threshold': threshold, 'Failure probability': None})
        return results
