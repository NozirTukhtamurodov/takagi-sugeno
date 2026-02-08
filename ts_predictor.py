"""
Лёгкий standalone-предиктор Такаги-Сугено.

Загружает модель из plain-dict pickle файла (сохранённого через
TakagiSugenoClassifier.save_model) и выполняет предсказания.

Зависимости: numpy, scipy (только scipy.special.softmax).

Использование:
    from ts_predictor import TakagiSugenoPredictor

    model = TakagiSugenoPredictor.load("model.pkl")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = model.classes_
"""

import pickle
from typing import Optional

import numpy as np
from scipy.special import softmax as scipy_softmax


class TakagiSugenoPredictor:
    """
    Standalone-предиктор Такаги-Сугено для использования в API.

    Совместим со scikit-learn интерфейсом:
      - predict(X) -> np.ndarray
      - predict_proba(X) -> np.ndarray
      - classes_ -> np.ndarray
    """

    def __init__(
        self,
        n_inputs: int,
        n_classes: int,
        n_mfs: int,
        temperature: float,
        use_neutrosophic: bool,
        partitions_centers: list[np.ndarray],
        partitions_sigmas: list[np.ndarray],
        rule_indices: np.ndarray,
        consequent_params: np.ndarray,
        neutrosophic_intervals: Optional[np.ndarray] = None,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
    ):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.n_mfs = n_mfs
        self.temperature = temperature
        self.use_neutrosophic = use_neutrosophic

        self.partitions_centers = partitions_centers
        self.partitions_sigmas = partitions_sigmas
        self.rule_indices = rule_indices
        self.consequent_params = consequent_params

        self.neutrosophic_intervals = neutrosophic_intervals

        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

        # scikit-learn совместимый атрибут
        self.classes_ = np.arange(n_classes)

    # ------------------------------------------------------------------
    # Загрузка из pickle-файла
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "TakagiSugenoPredictor":
        """Загрузка модели из pickle-файла (plain dict)."""
        with open(path, "rb") as f:
            d = pickle.load(f)

        partitions_centers = []
        partitions_sigmas = []
        for p in d["partitions"]:
            partitions_centers.append(np.array(p["centers"], dtype=np.float64))
            partitions_sigmas.append(np.array(p["sigmas"], dtype=np.float64))

        neutro = None
        if d.get("neutrosophic_intervals") is not None:
            neutro = np.array(d["neutrosophic_intervals"], dtype=np.float64)

        scaler_mean = None
        scaler_scale = None
        if d.get("scaler_mean") is not None:
            scaler_mean = np.array(d["scaler_mean"], dtype=np.float64)
            scaler_scale = np.array(d["scaler_scale"], dtype=np.float64)

        return cls(
            n_inputs=d["n_inputs"],
            n_classes=d["n_classes"],
            n_mfs=d["n_mfs"],
            temperature=d.get("temperature", 1.0),
            use_neutrosophic=d.get("use_neutrosophic", False),
            partitions_centers=partitions_centers,
            partitions_sigmas=partitions_sigmas,
            rule_indices=np.array(d["rule_indices"], dtype=np.int32),
            consequent_params=np.array(d["consequent_params"], dtype=np.float64),
            neutrosophic_intervals=neutro,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Вероятности классов.

        Args:
            X: (n_samples, n_inputs)

        Returns:
            (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = self._defuzzify(X)
        return scipy_softmax(scores / self.temperature, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток классов."""
        return np.argmax(self.predict_proba(X), axis=1)

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian(x: np.ndarray, centers: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """Гауссова функция принадлежности: (n_samples,) -> (n_samples, n_mfs)."""
        diff_sq = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
        return np.exp(-diff_sq / (2 * sigmas[np.newaxis, :] ** 2))

    def _compute_all_memberships(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_inputs) -> (n_samples, n_inputs, n_mfs)."""
        n_samples = X.shape[0]
        memberships = np.empty(
            (n_samples, self.n_inputs, self.n_mfs), dtype=np.float64
        )
        for i in range(self.n_inputs):
            memberships[:, i, :] = self._gaussian(
                X[:, i], self.partitions_centers[i], self.partitions_sigmas[i]
            )
        return memberships

    def _compute_firing(self, memberships: np.ndarray) -> np.ndarray:
        """Вычисление нормализованных сил срабатывания."""
        n_samples = memberships.shape[0]
        n_rules = len(self.rule_indices)
        n_inputs = memberships.shape[1]

        if self.use_neutrosophic and self.neutrosophic_intervals is not None:
            # --- Нейтрософская стратегия ---
            intervals = self.neutrosophic_intervals
            n_levels = len(intervals)
            levels = np.clip(
                np.round(memberships * (n_levels - 1)).astype(int),
                0, n_levels - 1,
            )
            T = (intervals[levels, 0] + intervals[levels, 1]) / 2

            firing = np.ones((n_samples, n_rules), dtype=np.float64)
            indeterminacy_sum = np.zeros((n_samples, n_rules), dtype=np.float64)

            I = (intervals[levels, 2] + intervals[levels, 3]) / 2

            for input_idx in range(n_inputs):
                mf_indices = self.rule_indices[:, input_idx]
                firing *= T[:, input_idx, mf_indices]
                indeterminacy_sum += I[:, input_idx, mf_indices]

            avg_indeterminacy = indeterminacy_sum / n_inputs
            confidence_factor = 1 - avg_indeterminacy * 0.5
            firing = firing * confidence_factor
        else:
            # --- Классическая стратегия ---
            firing = np.ones((n_samples, n_rules), dtype=np.float64)
            for input_idx in range(n_inputs):
                mf_indices = self.rule_indices[:, input_idx]
                firing *= memberships[:, input_idx, mf_indices]

        # Нормализация
        firing_sum = firing.sum(axis=1, keepdims=True) + 1e-8
        return firing / firing_sum

    def _compute_rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_inputs) -> (n_samples, n_rules, n_classes)."""
        n_samples = X.shape[0]
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        return np.einsum("sp,rcp->src", X_bias, self.consequent_params)

    def _defuzzify(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_inputs) -> (n_samples, n_classes)."""
        memberships = self._compute_all_memberships(X)
        normalized_firing = self._compute_firing(memberships)
        rule_outputs = self._compute_rule_outputs(X)
        return np.einsum("sr,src->sc", normalized_firing, rule_outputs)
