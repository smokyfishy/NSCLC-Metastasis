"""Denoising autoencoder for the RNA-seq (``rs_``) branch.

Architecture preserved from the notebooks (Final, cell 15; LASSO, cell 16)::

    Input(n_genes) -> Gaussian noise (training only) -> Dense(64, relu) -> Dense(32, relu)  [latent]
                                    -> Dense(64, relu) -> Dense(n_genes, linear)
    loss = MSE, optimizer = Adam, epochs = 50, batch_size = 32

Corrections (see docs/REPRODUCTION_AUDIT.md, section RS):

1. Noise is applied once, by the ``GaussianNoise`` layer during training. The
   original also pre-noised the inputs with ``K.random_normal`` *and* kept the
   ``GaussianNoise`` layer, so noise was applied twice.
2. The original ``noise_factor = 50`` on standardised (unit-variance) data
   gives a signal-to-noise ratio of roughly 1:2500; the reconstruction target
   is essentially unrecoverable. The corrected default is 0.5, the default of
   the notebook's own ``add_noise`` helper and the value used in the "Copy"
   notebook's multi-view experiment. ``configs/legacy_hyperparameters.yaml``
   keeps 50 for comparison.
3. Training no longer uses the held-out test set as ``validation_data``.
4. The downstream classifier consumes the 32-dimensional latent code produced
   by the encoder. The LASSO notebook instead took
   ``X_train_encoded.var(axis=0).argsort()[:100]`` — indices into the 32 latent
   units — and used them to slice the 401 *original gene columns*, so the
   "top features" were simply the first 32 gene columns in an arbitrary order.

TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def _keras():
    try:
        import tensorflow  # noqa: F401  (ensures the TF backend is installed)
        import keras  # Keras 3, shipped with TensorFlow >= 2.16
    except ImportError as exc:  # pragma: no cover - exercised only without TF
        raise ImportError(
            "TensorFlow is required for the RNA-seq autoencoder branch. "
            "Install with: python -m pip install -e \".[dev]\""
        ) from exc
    return keras


_NOISE_LAYER = None


def _noise_layer_cls():
    """Additive Gaussian noise active only in training, with no upper bound on
    the standard deviation.

    Keras 3's built-in ``GaussianNoise`` rejects ``stddev > 1``, so the
    original ``GaussianNoise(50)`` (valid in the Keras 2 used on Colab) cannot
    be built on current TensorFlow. This layer has the same semantics and lets
    ``configs/legacy_hyperparameters.yaml`` run.
    """
    global _NOISE_LAYER
    if _NOISE_LAYER is None:
        keras = _keras()

        @keras.saving.register_keras_serializable(package="nsclc_metastasis")
        class TrainingGaussianNoise(keras.layers.Layer):
            def __init__(self, stddev, seed=None, **kwargs):
                super().__init__(**kwargs)
                self.stddev = float(stddev)
                self.seed = seed
                self.seed_generator = keras.random.SeedGenerator(seed)

            def call(self, inputs, training=False):
                if training and self.stddev > 0:
                    return inputs + keras.random.normal(
                        keras.ops.shape(inputs), stddev=self.stddev, dtype=inputs.dtype,
                        seed=self.seed_generator,
                    )
                return inputs

            def compute_output_shape(self, input_shape):
                return input_shape

            def get_config(self):
                return {**super().get_config(), "stddev": self.stddev, "seed": self.seed}

        _NOISE_LAYER = TrainingGaussianNoise
    return _NOISE_LAYER


def build_autoencoder(n_features: int, hidden_units: int, latent_units: int, noise_stddev: float,
                      seed: int | None = None):
    keras = _keras()
    layers = keras.layers
    inp = keras.Input(shape=(n_features,), name="rs_input")
    x = _noise_layer_cls()(noise_stddev, seed=seed, name="gaussian_noise")(inp)
    x = layers.Dense(hidden_units, activation="relu", name="enc_hidden")(x)
    z = layers.Dense(latent_units, activation="relu", name="latent")(x)
    d = layers.Dense(hidden_units, activation="relu", name="dec_hidden")(z)
    out = layers.Dense(n_features, activation="linear", name="reconstruction")(d)
    autoencoder = keras.Model(inp, out, name="rs_denoising_autoencoder")
    encoder = keras.Model(inp, z, name="rs_encoder")
    return autoencoder, encoder


class DenoisingAutoencoder(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: fit a DAE, ``transform`` returns latent codes.

    Pickle-safe: Keras objects are dropped from the pickled state and the
    weights are stored as NumPy arrays, so a fitted pipeline containing this
    step can be saved with ``joblib``. The model is rebuilt on first use.
    """

    def __init__(
        self,
        hidden_units=64,
        latent_units=32,
        epochs=50,
        batch_size=32,
        noise_stddev=0.5,
        learning_rate=0.001,
        seed=39,
        verbose=0,
    ):
        self.hidden_units = hidden_units
        self.latent_units = latent_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_stddev = noise_stddev
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y=None):
        keras = _keras()
        X = np.asarray(X, dtype="float32")
        keras.utils.set_random_seed(int(self.seed))
        ae, enc = build_autoencoder(
            X.shape[1], self.hidden_units, self.latent_units, self.noise_stddev, seed=int(self.seed)
        )
        ae.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")
        hist = ae.fit(
            X,
            X,
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            shuffle=True,
            verbose=self.verbose,
        )
        self._autoencoder, self._encoder = ae, enc
        self.weights_ = [w.copy() for w in ae.get_weights()]
        self.n_features_in_ = X.shape[1]
        self.loss_history_ = [float(v) for v in hist.history["loss"]]
        return self

    def _ensure_model(self):
        if getattr(self, "_encoder", None) is None:
            ae, enc = build_autoencoder(
                self.n_features_in_, self.hidden_units, self.latent_units, self.noise_stddev, seed=int(self.seed)
            )
            ae.set_weights(self.weights_)
            self._autoencoder, self._encoder = ae, enc

    def transform(self, X):
        check_is_fitted(self, "weights_")
        X = np.asarray(X, dtype="float32")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        self._ensure_model()
        # Direct call (inference mode, noise off) avoids predict()'s per-model
        # tf.function retracing when many fold models are used in one process.
        return _keras().ops.convert_to_numpy(self._encoder(X, training=False))

    def reconstruct(self, X):
        check_is_fitted(self, "weights_")
        self._ensure_model()
        return _keras().ops.convert_to_numpy(self._autoencoder(np.asarray(X, dtype="float32"), training=False))

    def save_keras(self, path) -> None:
        """Export the full autoencoder as a native ``.keras`` file (for inspection)."""
        check_is_fitted(self, "weights_")
        self._ensure_model()
        self._autoencoder.save(str(path))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_autoencoder", None)
        state.pop("_encoder", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._autoencoder = None
        self._encoder = None
