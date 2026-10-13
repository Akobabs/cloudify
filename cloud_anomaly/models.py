"""Common scoring convention: larger scores always mean more anomalous."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import joblib

NAMES = {
    "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM",
    "random_forest": "Random Forest", "lstm_autoencoder": "LSTM Autoencoder",
}


class TabularDetector:
    def __init__(self, kind: str, parameters: dict, seed: int, threads: int):
        self.kind = kind
        self.threshold = None
        if kind == "isolation_forest":
            self.model = IsolationForest(**parameters, contamination="auto", random_state=seed, n_jobs=threads)
        elif kind == "one_class_svm":
            self.model = OneClassSVM(**parameters, kernel="rbf", cache_size=256)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(**parameters, class_weight="balanced",
                min_samples_leaf=2, random_state=seed, n_jobs=threads)
        else:
            raise ValueError(kind)

    def fit(self, x, y=None, validation=None):
        self.model.fit(x, y)
        return self

    def score(self, x):
        if self.kind == "random_forest":
            index = list(self.model.classes_).index(1)
            return self.model.predict_proba(x)[:, index]
        return -self.model.score_samples(x)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        joblib.dump(self, path.with_suffix(".joblib"))


class LSTMAutoencoder:
    def __init__(self, parameters, seed, threads, shape):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass  # Thread settings are process-wide and already set for subsequent candidates.
        self.parameters = parameters
        self.threshold = None
        width = parameters["latent_dim"]
        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(width)(x)
        x = tf.keras.layers.RepeatVector(shape[0])(x)
        x = tf.keras.layers.LSTM(width, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))(x)
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.history = {}

    def fit(self, x, y=None, validation=None):
        tf = self.tf
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        def dataset(values):
            return tf.data.Dataset.from_tensor_slices((values, values)).batch(
                self.parameters["batch_size"]).with_options(options)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                     restore_best_weights=True)]
        history = self.model.fit(dataset(x), validation_data=dataset(validation),
            epochs=self.parameters["epochs"], callbacks=callbacks, verbose=0, shuffle=False)
        self.history = {key: [float(v) for v in values] for key, values in history.history.items()}
        return self

    def score(self, x):
        # Last-step error produces a point score aligned with each window endpoint.
        scores = []
        for start in range(0, len(x), 256):
            batch = x[start:start + 256]
            reconstruction = self.model(batch, training=False).numpy()
            scores.append(np.mean(np.square(batch[:, -1, :] - reconstruction[:, -1, :]), axis=1))
        return np.concatenate(scores)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        self.model.save(path.with_suffix(".keras"))
