"""
Оптимизированный классификатор Такаги-Сугено для многоклассовой классификации

Полностью оптимизированная реализация с:
- Векторизованными операциями NumPy (без ручных циклов где возможно)
- SciPy для линейной алгебры и специальных функций
- Принципами SOLID (Единственная ответственность, Открытость/Закрытость, Инверсия зависимостей)
- Принципом DRY (Не повторяйся)

Оптимизации производительности:
- Векторизованное вычисление принадлежностей
- Пакетное вычисление силы срабатывания правил
- scipy.linalg.solve для быстрого решения линейных систем
- scipy.special.softmax для численно устойчивого softmax
- Предварительно вычисленные таблицы поиска где применимо

Автор: Классификатор Такаги-Сугено (Оптимизированный)
Дата: Декабрь 2024
"""

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.special import softmax as scipy_softmax
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve, auc, precision_recall_curve
)
import joblib
import pickle
from typing import Tuple, Optional, Dict, List, Protocol, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import seaborn as sns
from itertools import cycle

# Подавление предупреждений
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')

# ============================================================================
# КОНСТАНТЫ
# ============================================================================

# Интервалы нейтрософских множеств по умолчанию (статические)
# Формат: [T_min, T_max, I_min, I_max, F_min, F_max]
# T - истинность, I - неопределённость, F - ложность
DEFAULT_NEUTROSOPHIC_INTERVALS = np.array([
    [0.000, 0.050, 0.050, 0.100, 0.850, 0.950],  # Уровень 0 - Очень низкий
    [0.050, 0.150, 0.080, 0.120, 0.730, 0.870],  # Уровень 1 - Низкий
    [0.150, 0.275, 0.100, 0.150, 0.575, 0.750],  # Уровень 2 - Ниже среднего
    [0.275, 0.400, 0.130, 0.180, 0.420, 0.595],  # Уровень 3 - Средне-низкий
    [0.400, 0.500, 0.150, 0.200, 0.300, 0.450],  # Уровень 4 - Средний
    [0.500, 0.625, 0.130, 0.180, 0.195, 0.370],  # Уровень 5 - Средне-высокий
    [0.625, 0.750, 0.100, 0.150, 0.100, 0.275],  # Уровень 6 - Выше среднего
    [0.750, 0.875, 0.080, 0.120, 0.005, 0.170],  # Уровень 7 - Высокий
    [0.875, 1.000, 0.050, 0.100, 0.000, 0.075],  # Уровень 8 - Очень высокий
], dtype=np.float64)


# ============================================================================
# АБСТРАКТНЫЕ ИНТЕРФЕЙСЫ (SOLID - Инверсия зависимостей)
# ============================================================================

class MembershipFunction(Protocol):
    """Протокол для функций принадлежности (SOLID - Принцип открытости/закрытости)."""
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
    @property
    def center(self) -> float: ...
    @property
    def sigma(self) -> float: ...


class FuzzyInferenceStrategy(ABC):
    """Абстрактная стратегия нечёткого вывода (SOLID - Открытость/Закрытость)."""
    
    @abstractmethod
    def compute_firing_strengths(
        self, 
        memberships: np.ndarray, 
        rule_indices: np.ndarray
    ) -> np.ndarray:
        """Вычислить силы срабатывания правил."""
        pass


# ============================================================================
# КЛАССЫ ДАННЫХ (Неизменяемая конфигурация)
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Неизменяемая конфигурация классификатора (SOLID - Единственная ответственность)."""
    n_inputs: int = 17            # Количество входных признаков
    n_classes: int = 5            # Количество классов
    n_mfs: int = 5                # Количество функций принадлежности
    regularization: float = 0.01  # Коэффициент регуляризации
    temperature: float = 1.0      # Температура softmax
    use_neutrosophic: bool = False     # Использовать нейтрософскую логику
    dynamic_neutrosophic: bool = True  # Динамические интервалы
    max_rules: int = 150          # Максимальное количество правил
    overlap_factor: float = 1.5   # Коэффициент перекрытия ФП


@dataclass
class NeutrosophicConfig:
    """Конфигурация нейтрософской логики."""
    n_levels: int = 9                    # 9 уровней термов (0-8)
    indeterminacy_factor: float = 0.05  # Фактор неопределённости
    falsity_factor: float = 0.1         # Фактор ложности
    intervals: Optional[np.ndarray] = None        # Интервалы
    quantile_boundaries: Optional[np.ndarray] = None  # Границы квантилей


@dataclass
class HyperConfig:
    """
    Централизованная конфигурация гиперпараметров.
    Все гиперпараметры в одном месте для удобства настройки.
    """
    # Параметры функций принадлежности
    n_mfs_min: int = 5              # Минимальное количество ФП
    n_mfs_max: int = 7              # Максимальное количество ФП
    n_mfs_divisor: int = 20         # Делитель для адаптивного расчёта n_mfs
    
    # Параметры правил
    max_rules_min: int = 200        # Минимальное количество правил
    max_rules_max: int = 350        # Максимальное количество правил  
    max_rules_multiplier: int = 2   # Множитель классов для расчёта правил
    
    # Регуляризация
    regularization: float = 0.01    # Коэффициент регуляризации
    
    # Температура softmax
    temperature: float = 0.5        # Температура (меньше = резче решения)
    
    # Перекрытие функций принадлежности
    overlap_factor: float = 2.0     # Коэффициент перекрытия ФП
    
    # Порог для "многоклассовой" задачи
    multiclass_threshold: int = 50  # Порог количества классов
    
    # === ПАРАМЕТРЫ PCA (отбор признаков) ===
    use_pca: bool = True            # Использовать PCA для уменьшения размерности
    pca_variance: float = 0.95      # Доля объяснённой дисперсии (0.95 = 95%)
    pca_n_components: Optional[int] = None  # Фиксированное число компонент (если задано, игнорирует pca_variance)
    
    # === ПАРАМЕТРЫ РАЗДЕЛЕНИЯ ДАННЫХ ===
    test_size: float = 0.1          # Доля тестовой выборки (0.1 = 10%)
    random_state: int = 42          # Seed для воспроизводимости
    
    # === ПАРАМЕТРЫ АНСАМБЛЯ ===
    use_ensemble: bool = False      # Отключить ансамбль
    n_estimators: int = 5           # Количество моделей в ансамбле
    ensemble_diversity: bool = True # Разнообразие через разные параметры
    
    # === ИЕРАРХИЧЕСКАЯ КЛАССИФИКАЦИЯ ===
    use_hierarchical: bool = False  # Отключить иерархическую классификацию
    n_groups: int = 10              # Количество групп (кластеров) на первом уровне
    
    # === БУСТИНГ (АДАПТИВНОЕ ОБУЧЕНИЕ) ===
    use_boosting: bool = True       # Использовать бустинг
    n_boosting_rounds: int = 5      # Количество раундов бустинга
    boosting_learning_rate: float = 0.3  # Скорость обучения бустинга
    
    def calculate_adaptive_params(self, n_classes: int, n_features: int) -> dict:
        """
        Рассчитать адаптивные параметры на основе данных.
        
        Args:
            n_classes: Количество классов
            n_features: Количество признаков
            
        Returns:
            dict с параметрами: n_mfs, max_rules, regularization, temperature, overlap_factor
        """
        # Адаптивное количество ФП
        n_mfs = min(self.n_mfs_max, max(self.n_mfs_min, n_classes // self.n_mfs_divisor))
        
        # Адаптивное количество правил
        max_rules = min(self.max_rules_max, max(self.max_rules_min, n_classes * self.max_rules_multiplier))
        
        return {
            'n_mfs': n_mfs,
            'max_rules': max_rules,
            'regularization': self.regularization,
            'temperature': self.temperature,
            'overlap_factor': self.overlap_factor
        }


# Глобальная конфигурация гиперпараметров (изменяйте здесь для настройки)
HYPER_CONFIG = HyperConfig(
    n_mfs_min=3,
    n_mfs_max=5,            # 3-5 ФП достаточно для малого числа классов
    n_mfs_divisor=20,       # Стандартный делитель
    max_rules_min=30,       # Разумный минимум для 4 классов
    max_rules_max=100,      # Разумный максимум (не больше кол-ва образцов)
    max_rules_multiplier=10, # ~40 правил для 4 классов
    regularization=0.01,    # Стандартная регуляризация
    temperature=0.5,        # Умеренная температура softmax
    overlap_factor=1.5,     # Стандартное перекрытие ФП
    multiclass_threshold=50,
    # PCA параметры
    use_pca=False,          # Отключить PCA - сохраняем все признаки
    pca_variance=0.95,      # 95% дисперсии
    pca_n_components=None,  # Авто-выбор компонент
    # Разделение данных
    test_size=0.2,          # 20% тестовая выборка (80 образцов — надёжнее)
    random_state=42,
    # Ансамбль
    use_ensemble=False,     # Отключить ансамбль
    n_estimators=5,
    ensemble_diversity=True,
    # Иерархическая классификация
    use_hierarchical=False, # Отключить иерархическую классификацию
    n_groups=10,            # 10 групп на первом уровне
    # Бустинг
    use_boosting=False,     # Отключить бустинг (не нужен для 4 классов)
    n_boosting_rounds=5,    # Количество раундов бустинга
    boosting_learning_rate=0.3  # Скорость обучения бустинга
)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (DRY - Повторно используемые функции)
# ============================================================================

def gaussian_membership_vectorized(
    x: np.ndarray, 
    centers: np.ndarray, 
    sigmas: np.ndarray
) -> np.ndarray:
    """
    Векторизованное вычисление гауссовой принадлежности.
    
    Аргументы:
        x: Входные значения (n_samples,)
        centers: Центры ФП (n_mfs,)
        sigmas: Стандартные отклонения ФП (n_mfs,)
        
    Возвращает:
        Степени принадлежности (n_samples, n_mfs)
    """
    # Broadcasting: x[:, None] -> (n_samples, 1), centers -> (n_mfs,)
    # Результат: (n_samples, n_mfs)
    diff_sq = (x[:, np.newaxis] - centers[np.newaxis, :]) ** 2
    return np.exp(-diff_sq / (2 * sigmas[np.newaxis, :] ** 2))


def generate_dynamic_intervals(
    data: np.ndarray,
    n_levels: int = 9,
    indeterminacy_factor: float = 0.05,
    falsity_factor: float = 0.1
) -> np.ndarray:
    """
    Генерация нейтрософских интервалов из квантилей данных (векторизованно).
    
    Аргументы:
        data: Массив входных данных
        n_levels: Количество дискретных уровней (9 по умолчанию)
        indeterminacy_factor: Фактор для вычисления неопределённости
        falsity_factor: Фактор для вычисления ложности
        
    Возвращает:
        Массив интервалов размером (n_levels, 6)
    """
    flat_data = data.ravel()
    quantiles = np.linspace(0, 100, n_levels + 1)
    boundaries = np.percentile(flat_data, quantiles)
    
    # Нормализация в [0, 1]
    data_min, data_max = flat_data.min(), flat_data.max()
    data_range = max(data_max - data_min, 1e-8)
    boundaries_norm = np.clip((boundaries - data_min) / data_range, 0, 1)
    
    intervals = np.zeros((n_levels, 6), dtype=np.float64)
    
    # Интервалы истинности
    intervals[:, 0] = boundaries_norm[:-1]  # T_min
    intervals[:, 1] = boundaries_norm[1:]   # T_max
    
    # Факторы уровней для неопределённости (векторизованно)
    level_factors = 1.0 - np.arange(n_levels) / (n_levels - 1) * 0.5
    
    # Интервалы неопределённости
    intervals[:, 2] = intervals[:, 0] * indeterminacy_factor * level_factors
    intervals[:, 3] = intervals[:, 1] * indeterminacy_factor * level_factors * 1.5
    
    # Интервалы ложности
    intervals[:, 4] = np.maximum(0, (1 - intervals[:, 1]) * falsity_factor)
    intervals[:, 5] = np.maximum(0, (1 - intervals[:, 0]) * falsity_factor * 1.5)
    
    return intervals


def one_hot_encode(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Векторизованное one-hot кодирование с защитой от выхода за границы."""
    n_samples = len(y)
    y_int = y.astype(int)
    
    # Защита: проверяем границы
    max_idx = y_int.max()
    if max_idx >= n_classes:
        raise ValueError(
            f"Ошибка one_hot_encode: max(y)={max_idx} >= n_classes={n_classes}. "
            f"Проверьте переиндексацию классов! Уникальные y: {np.unique(y_int)[:20]}..."
        )
    
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
    one_hot[np.arange(n_samples), y_int] = 1.0
    return one_hot


def solve_ridge_regression(
    Phi: np.ndarray, 
    Y: np.ndarray, 
    regularization: float
) -> np.ndarray:
    """
    Решение ридж-регрессии с использованием scipy.linalg.
    
    P = (Φ'Φ + λI)^(-1) Φ' Y
    
    Аргументы:
        Phi: Матрица плана (n_samples, n_features)
        Y: Матрица целей (n_samples, n_outputs)
        regularization: Сила регуляризации
        
    Возвращает:
        Параметры (n_features, n_outputs)
    """
    PhiT_Phi = Phi.T @ Phi
    PhiT_Y = Phi.T @ Y
    reg = regularization * np.eye(PhiT_Phi.shape[0])
    
    try:
        # Использование scipy.linalg.solve (быстрее и стабильнее np.linalg.solve)
        return linalg.solve(PhiT_Phi + reg, PhiT_Y, assume_a='pos')
    except linalg.LinAlgError:
        # Запасной вариант - lstsq с регуляризацией
        return linalg.lstsq(Phi, Y, cond=regularization)[0]


# ============================================================================
# ОБРАБОТЧИК НЕЙТРОСОФСКОЙ ЛОГИКИ (SOLID - Единственная ответственность)
# ============================================================================

class NeutrosophicHandler:
    """
    Обработчик нейтрософских вычислений принадлежности.
    
    Отвечает только за операции нейтрософской логики.
    """
    
    def __init__(self, config: NeutrosophicConfig):
        self.config = config
        if config.intervals is not None:
            self.intervals = config.intervals
        else:
            self.intervals = DEFAULT_NEUTROSOPHIC_INTERVALS.copy()
    
    @classmethod
    def from_data(cls, data: np.ndarray, n_levels: int = 9) -> 'NeutrosophicHandler':
        """Фабричный метод для создания обработчика с динамическими интервалами из данных."""
        intervals = generate_dynamic_intervals(data, n_levels)
        config = NeutrosophicConfig(n_levels=n_levels, intervals=intervals)
        return cls(config)
    
    def get_neutrosophic_values(self, memberships: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Векторизованное преобразование принадлежностей в нейтрософские тройки.
        
        Аргументы:
            memberships: Классические значения принадлежности (любая форма)
            
        Возвращает:
            Кортеж (T, I, F) массивов той же формы
        """
        # Отображение принадлежностей на дискретные уровни
        n_levels = len(self.intervals)
        levels = np.clip(
            np.round(memberships * (n_levels - 1)).astype(int),
            0, n_levels - 1
        )
        
        # Поиск нейтрософских значений (векторизованно)
        # Средние точки интервалов
        T = (self.intervals[levels, 0] + self.intervals[levels, 1]) / 2
        I = (self.intervals[levels, 2] + self.intervals[levels, 3]) / 2
        F = (self.intervals[levels, 4] + self.intervals[levels, 5]) / 2
        
        return T, I, F
    
    def update_intervals(self, data: np.ndarray) -> None:
        """Обновление интервалов из новых данных."""
        self.intervals = generate_dynamic_intervals(
            data, 
            self.config.n_levels,
            self.config.indeterminacy_factor,
            self.config.falsity_factor
        )


# ============================================================================
# НЕЧЁТКОЕ РАЗБИЕНИЕ (SOLID - Единственная ответственность)
# ============================================================================

class FuzzyPartition:
    """
    Векторизованное нечёткое разбиение для одной входной переменной.
    
    Хранит параметры как массивы NumPy для векторизованных операций.
    """
    
    def __init__(
        self,
        n_mfs: int,
        min_val: float,
        max_val: float,
        overlap_factor: float = 1.5
    ):
        self.n_mfs = n_mfs
        self.min_val = min_val
        self.max_val = max_val
        
        # Вычисление центров и сигм как массивов (DRY - однократное вычисление)
        self.centers = np.linspace(min_val, max_val, n_mfs)
        base_spread = (max_val - min_val) / (2 * max(n_mfs - 1, 1))
        self.sigmas = np.full(n_mfs, base_spread * overlap_factor)
    
    def fuzzify(self, x: np.ndarray) -> np.ndarray:
        """
        Векторизованная фаззификация.
        
        Аргументы:
            x: Входные значения (n_samples,)
            
        Возвращает:
            Степени принадлежности (n_samples, n_mfs)
        """
        return gaussian_membership_vectorized(x, self.centers, self.sigmas)
    
    def to_dict(self) -> dict:
        """Сериализация в словарь."""
        return {
            'n_mfs': self.n_mfs,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'centers': self.centers.tolist(),
            'sigmas': self.sigmas.tolist()
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'FuzzyPartition':
        """Десериализация из словаря."""
        partition = cls(d['n_mfs'], d['min_val'], d['max_val'], overlap_factor=1.0)
        partition.centers = np.array(d['centers'])
        partition.sigmas = np.array(d['sigmas'])
        return partition


# ============================================================================
# ГЕНЕРАТОР ПРАВИЛ (SOLID - Единственная ответственность)
# ============================================================================

class RuleGenerator:
    """
    Генерация нечётких правил из данных.
    
    Отвечает только за логику генерации правил.
    """
    
    @staticmethod
    def generate_class_balanced_rules(
        memberships: np.ndarray,
        y: np.ndarray,
        n_mfs: int,
        n_inputs: int,
        max_rules: int = 100
    ) -> np.ndarray:
        """
        Генерация сбалансированных по классам правил из паттернов данных.
        
        Аргументы:
            memberships: Степени принадлежности (n_samples, n_inputs, n_mfs)
            y: Метки классов (n_samples,)
            n_mfs: Количество функций принадлежности
            n_inputs: Количество входных признаков
            max_rules: Максимальное количество правил
            
        Возвращает:
            Массив индексов правил (n_rules, n_inputs)
        """
        rule_set = set()
        
        # 1. Добавление диагональных правил
        for mf_idx in range(n_mfs):
            rule_set.add(tuple([mf_idx] * n_inputs))
        
        # 2. Добавление сбалансированных по классам правил из данных
        unique_classes = np.unique(y)
        rules_per_class = max(max_rules // len(unique_classes), 10)
        
        # Векторизованно: получаем победившую ФП для каждого образца и входа
        winning_mfs = np.argmax(memberships, axis=2)  # (n_samples, n_inputs)
        
        for class_label in unique_classes:
            class_mask = (y == class_label)
            class_winning_mfs = winning_mfs[class_mask]  # (n_class_samples, n_inputs)
            
            # Преобразование в кортежи и добавление в множество
            for i in range(min(len(class_winning_mfs), rules_per_class)):
                rule_tuple = tuple(class_winning_mfs[i])
                rule_set.add(rule_tuple)
                
                # Добавление соседних правил (только для первых 5 входов для эффективности)
                if len(rule_set) < max_rules:
                    rule = list(rule_tuple)
                    for input_idx in range(min(5, n_inputs)):
                        for delta in [-1, 1]:
                            new_mf = rule[input_idx] + delta
                            if 0 <= new_mf < n_mfs:
                                neighbor = rule.copy()
                                neighbor[input_idx] = new_mf
                                rule_set.add(tuple(neighbor))
                
                if len(rule_set) >= max_rules:
                    break
            
            if len(rule_set) >= max_rules:
                break
        
        rules = list(rule_set)[:max_rules]
        return np.array(rules, dtype=np.int32)


# ============================================================================
# СТРАТЕГИИ ВЫЧИСЛЕНИЯ СИЛЫ СРАБАТЫВАНИЯ (SOLID - Открытость/Закрытость)
# ============================================================================

class ClassicalFiringStrategy(FuzzyInferenceStrategy):
    """Классическое вычисление силы срабатывания нечётких правил."""
    
    def compute_firing_strengths(
        self,
        memberships: np.ndarray,
        rule_indices: np.ndarray
    ) -> np.ndarray:
        """
        Векторизованное вычисление силы срабатывания.
        
        Аргументы:
            memberships: (n_samples, n_inputs, n_mfs)
            rule_indices: (n_rules, n_inputs)
            
        Возвращает:
            Силы срабатывания (n_samples, n_rules)
        """
        n_samples = memberships.shape[0]
        n_rules = len(rule_indices)
        n_inputs = memberships.shape[1]
        
        # Инициализация единицами (для произведения t-нормы)
        firing = np.ones((n_samples, n_rules), dtype=np.float64)
        
        # Векторизованное произведение по входам
        for input_idx in range(n_inputs):
            mf_indices = rule_indices[:, input_idx]  # (n_rules,)
            # memberships[:, input_idx, :] -> (n_samples, n_mfs)
            # Индексация по mf_indices для получения (n_samples, n_rules)
            firing *= memberships[:, input_idx, mf_indices]
        
        return firing


class NeutrosophicFiringStrategy(FuzzyInferenceStrategy):
    """Нейтрософское вычисление силы срабатывания нечётких правил."""
    
    def __init__(self, handler: NeutrosophicHandler):
        self.handler = handler
        self.last_indeterminacy: Optional[np.ndarray] = None
    
    def compute_firing_strengths(
        self,
        memberships: np.ndarray,
        rule_indices: np.ndarray
    ) -> np.ndarray:
        """
        Нейтрософская сила срабатывания с корректировкой неопределённости.
        
        Аргументы:
            memberships: (n_samples, n_inputs, n_mfs)
            rule_indices: (n_rules, n_inputs)
            
        Возвращает:
            Силы срабатывания (n_samples, n_rules)
        """
        n_samples, n_inputs, n_mfs = memberships.shape
        n_rules = len(rule_indices)
        
        # Преобразование в нейтрософские значения
        T, I, F = self.handler.get_neutrosophic_values(memberships)
        
        # Инициализация сил срабатывания
        firing = np.ones((n_samples, n_rules), dtype=np.float64)
        indeterminacy_sum = np.zeros((n_samples, n_rules), dtype=np.float64)
        
        # Векторизованное произведение по входам
        for input_idx in range(n_inputs):
            mf_indices = rule_indices[:, input_idx]
            firing *= T[:, input_idx, mf_indices]
            indeterminacy_sum += I[:, input_idx, mf_indices]
        
        # Корректировка силы срабатывания по неопределённости
        avg_indeterminacy = indeterminacy_sum / n_inputs
        confidence_factor = 1 - avg_indeterminacy * 0.5
        
        self.last_indeterminacy = avg_indeterminacy
        
        return firing * confidence_factor


# ============================================================================
# ГЛАВНЫЙ КЛАССИФИКАТОР (SOLID - Композиция вместо наследования)
# ============================================================================

class TakagiSugenoClassifier:
    """
    Оптимизированный нечёткий классификатор Такаги-Сугено.
    
    Использует композицию и паттерн стратегии для гибкости.
    Все тяжёлые вычисления векторизованы с NumPy/SciPy.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        """
        Инициализация классификатора.
        
        Args:
            config: ModelConfig instance, or pass individual parameters as kwargs
        """
        if config is None:
            config = ModelConfig(**kwargs)
        self.config = config
        
        # Components (initialized during fit)
        self.partitions: List[FuzzyPartition] = []
        self.rule_indices: Optional[np.ndarray] = None
        self.consequent_params: Optional[np.ndarray] = None
        self.firing_strategy: Optional[FuzzyInferenceStrategy] = None
        self.neutrosophic_handler: Optional[NeutrosophicHandler] = None
        
        # Data bounds
        self.mins: Optional[np.ndarray] = None
        self.maxs: Optional[np.ndarray] = None
        self.scaler: Optional[StandardScaler] = None
        
        self.is_fitted = False
        self._actual_n_inputs = config.n_inputs
    
    def _create_partitions(self) -> None:
        """Create fuzzy partitions for all inputs."""
        self.partitions = []
        
        for i in range(self._actual_n_inputs):
            # Extend range for better coverage
            range_val = self.maxs[i] - self.mins[i]
            partition = FuzzyPartition(
                n_mfs=self.config.n_mfs,
                min_val=self.mins[i] - 0.1 * range_val,
                max_val=self.maxs[i] + 0.1 * range_val,
                overlap_factor=self.config.overlap_factor
            )
            self.partitions.append(partition)
    
    def _compute_all_memberships(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized membership computation for all inputs.
        
        Args:
            X: (n_samples, n_inputs)
            
        Returns:
            (n_samples, n_inputs, n_mfs)
        """
        n_samples = X.shape[0]
        memberships = np.empty(
            (n_samples, self._actual_n_inputs, self.config.n_mfs), 
            dtype=np.float64
        )
        
        for i, partition in enumerate(self.partitions):
            memberships[:, i, :] = partition.fuzzify(X[:, i])
        
        return memberships
    
    def _compute_normalized_firing(self, memberships: np.ndarray) -> np.ndarray:
        """Compute normalized firing strengths."""
        firing = self.firing_strategy.compute_firing_strengths(
            memberships, self.rule_indices
        )
        firing_sum = firing.sum(axis=1, keepdims=True) + 1e-8
        return firing / firing_sum
    
    def _compute_weighted_inputs(self, X: np.ndarray, memberships: np.ndarray) -> np.ndarray:
        """
        Compute weighted input matrix for Least Squares.
        
        Args:
            X: (n_samples, n_inputs)
            memberships: (n_samples, n_inputs, n_mfs)
            
        Returns:
            (n_samples, n_rules * (n_inputs + 1))
        """
        n_samples = X.shape[0]
        n_rules = len(self.rule_indices)
        n_params = self._actual_n_inputs + 1
        
        normalized_firing = self._compute_normalized_firing(memberships)
        
        # Add bias
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        # Vectorized construction of weighted inputs
        # Shape: (n_samples, n_rules, n_params)
        weighted = normalized_firing[:, :, np.newaxis] * X_bias[:, np.newaxis, :]
        
        # Reshape to (n_samples, n_rules * n_params)
        return weighted.reshape(n_samples, -1)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TakagiSugenoClassifier':
        """
        Fit the classifier.
        
        Args:
            X: Training features (n_samples, n_inputs)
            y: Training labels (n_samples,)
            
        Returns:
            self
        """
        # Store data bounds
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)
        self._actual_n_inputs = X.shape[1]
        
        # Инициализация обработчика нейтрософской логики
        if self.config.use_neutrosophic:
            print("    Генерация нейтрософских интервалов...")
            if self.config.dynamic_neutrosophic:
                self.neutrosophic_handler = NeutrosophicHandler.from_data(
                    X, n_levels=9  # 9 уровней нейтрософских термов (0-8)
                )
            else:
                self.neutrosophic_handler = NeutrosophicHandler(NeutrosophicConfig())
            self.firing_strategy = NeutrosophicFiringStrategy(self.neutrosophic_handler)
        else:
            self.firing_strategy = ClassicalFiringStrategy()
        
        # Create partitions
        print(f"    Creating {self.config.n_mfs} membership functions per input...")
        self._create_partitions()
        
        # Compute memberships once
        memberships = self._compute_all_memberships(X)
        
        # Generate rules
        print(f"    Генерация правил (макс. {self.config.max_rules})...")
        self.rule_indices = RuleGenerator.generate_class_balanced_rules(
            memberships, y, self.config.n_mfs, self._actual_n_inputs, self.config.max_rules
        )
        print(f"    Сгенерировано {len(self.rule_indices)} правил")
        
        # Compute weighted inputs
        print("    Оценка параметров консеквента...")
        Phi = self._compute_weighted_inputs(X, memberships)
        
        # One-hot encode targets
        Y = one_hot_encode(y, self.config.n_classes)
        
        # Solve ridge regression
        P_flat = solve_ridge_regression(Phi, Y, self.config.regularization)
        
        # Reshape parameters: (n_rules * (n_inputs + 1), n_classes) -> (n_rules, n_classes, n_inputs + 1)
        n_rules = len(self.rule_indices)
        n_params = self._actual_n_inputs + 1
        self.consequent_params = P_flat.reshape(n_rules, n_params, self.config.n_classes)
        self.consequent_params = np.transpose(self.consequent_params, (0, 2, 1))
        
        self.is_fitted = True
        return self
    
    def _compute_rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized rule output computation.
        
        Args:
            X: (n_samples, n_inputs)
            
        Returns:
            (n_samples, n_rules, n_classes)
        """
        n_samples = X.shape[0]
        X_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        # consequent_params: (n_rules, n_classes, n_inputs + 1)
        # X_bias: (n_samples, n_inputs + 1)
        # Result: (n_samples, n_rules, n_classes)
        return np.einsum('sp,rcp->src', X_bias, self.consequent_params)
    
    def _defuzzify(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized defuzzification.
        
        Args:
            X: (n_samples, n_inputs)
            
        Returns:
            (n_samples, n_classes)
        """
        memberships = self._compute_all_memberships(X)
        normalized_firing = self._compute_normalized_firing(memberships)
        rule_outputs = self._compute_rule_outputs(X)
        
        # Weighted sum: (n_samples, n_rules) @ (n_samples, n_rules, n_classes)
        # Using einsum for clarity
        return np.einsum('sr,src->sc', normalized_firing, rule_outputs)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using scipy.softmax.
        
        Args:
            X: (n_samples, n_inputs)
            
        Returns:
            (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Модель должна быть обучена.")
        
        scores = self._defuzzify(X)
        return scipy_softmax(scores / self.config.temperature, axis=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание меток классов."""
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Predict with uncertainty information.
        
        Returns:
            (predictions, probabilities, indeterminacy or None)
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        
        indeterminacy = None
        if isinstance(self.firing_strategy, NeutrosophicFiringStrategy):
            memberships = self._compute_all_memberships(X)
            normalized_firing = self._compute_normalized_firing(memberships)
            indeterminacy = np.sum(
                self.firing_strategy.last_indeterminacy * normalized_firing, axis=1
            )
        
        return predictions, proba, indeterminacy
    
    def get_prediction_confidence(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get detailed confidence metrics."""
        proba = self.predict_proba(X)
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]
        
        confidence = {
            'probability': proba.max(axis=1),
            'margin': sorted_proba[:, 0] - sorted_proba[:, 1],
            'entropy': -np.sum(proba * np.log(proba + 1e-10), axis=1)
        }
        
        if isinstance(self.firing_strategy, NeutrosophicFiringStrategy):
            memberships = self._compute_all_memberships(X)
            normalized_firing = self._compute_normalized_firing(memberships)
            confidence['indeterminacy'] = np.sum(
                self.firing_strategy.last_indeterminacy * normalized_firing, axis=1
            )
            confidence['neutrosophic_confidence'] = (
                confidence['probability'] * (1 - confidence['indeterminacy'])
            )
        
        return confidence
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return accuracy_score(y, self.predict(X))
    
    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_folds: int = 5
    ) -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accuracies, f1_scores_list = [], []
        
        for train_idx, val_idx in skf.split(X, y):
            fold_model = TakagiSugenoClassifier(config=self.config)
            
            # Silent fit
            fold_model.mins = X[train_idx].min(axis=0)
            fold_model.maxs = X[train_idx].max(axis=0)
            fold_model._actual_n_inputs = X.shape[1]
            
            if self.config.use_neutrosophic:
                if self.config.dynamic_neutrosophic:
                    fold_model.neutrosophic_handler = NeutrosophicHandler.from_data(X[train_idx])
                else:
                    fold_model.neutrosophic_handler = NeutrosophicHandler(NeutrosophicConfig())
                fold_model.firing_strategy = NeutrosophicFiringStrategy(fold_model.neutrosophic_handler)
            else:
                fold_model.firing_strategy = ClassicalFiringStrategy()
            
            fold_model._create_partitions()
            memberships = fold_model._compute_all_memberships(X[train_idx])
            fold_model.rule_indices = RuleGenerator.generate_class_balanced_rules(
                memberships, y[train_idx], self.config.n_mfs, 
                X.shape[1], self.config.max_rules
            )
            
            Phi = fold_model._compute_weighted_inputs(X[train_idx], memberships)
            Y = one_hot_encode(y[train_idx], self.config.n_classes)
            P_flat = solve_ridge_regression(Phi, Y, self.config.regularization)
            
            n_rules = len(fold_model.rule_indices)
            n_params = fold_model._actual_n_inputs + 1
            fold_model.consequent_params = P_flat.reshape(n_rules, n_params, self.config.n_classes)
            fold_model.consequent_params = np.transpose(fold_model.consequent_params, (0, 2, 1))
            fold_model.is_fitted = True
            
            y_pred = fold_model.predict(X[val_idx])
            accuracies.append(accuracy_score(y[val_idx], y_pred))
            f1_scores_list.append(f1_score(y[val_idx], y_pred, average='weighted'))
        
        return {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'f1_mean': np.mean(f1_scores_list),
            'f1_std': np.std(f1_scores_list),
            'fold_accuracies': accuracies
        }
    
    def get_rules_description(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """Get human-readable rule descriptions."""
        if feature_names is None:
            feature_names = [f"X{i}" for i in range(self._actual_n_inputs)]
        
        mf_labels = ["VeryLow", "Low", "Medium", "High", "VeryHigh", "ExtraHigh", "Max"][:self.config.n_mfs]
        descriptions = []
        
        for idx, rule in enumerate(self.rule_indices):
            parts = [f"{feature_names[i]} is {mf_labels[mf]}" 
                     for i, mf in enumerate(rule[:3])]
            antecedent = " AND ".join(parts)
            if len(rule) > 3:
                antecedent += f" AND ... ({len(rule) - 3} more)"
            descriptions.append(f"Rule {idx + 1}: IF {antecedent} THEN linear_output")
        
        return descriptions
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели в файл (чистый pickle формат).
        
        Сохраняется plain-словарь с numpy-массивами и базовыми типами,
        без ссылок на пользовательские классы — файл можно загрузить
        в любом окружении (API-сервер и т.д.) без модуля takagi_sugeno_optimized.
        """
        if not self.is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")
        
        # Сериализация partitions в списки (plain data)
        partitions_data = []
        for p in self.partitions:
            partitions_data.append({
                'n_mfs': int(p.n_mfs),
                'min_val': float(p.min_val),
                'max_val': float(p.max_val),
                'centers': p.centers.tolist(),
                'sigmas': p.sigmas.tolist(),
            })
        
        # Нейтрософские интервалы (если есть)
        neutrosophic_intervals = None
        use_neutrosophic = self.config.use_neutrosophic
        if use_neutrosophic and self.neutrosophic_handler is not None:
            neutrosophic_intervals = self.neutrosophic_handler.intervals.tolist()
        
        save_dict = {
            # Конфигурация
            'n_inputs': int(self._actual_n_inputs),
            'n_classes': int(self.config.n_classes),
            'n_mfs': int(self.config.n_mfs),
            'temperature': float(self.config.temperature),
            'use_neutrosophic': bool(use_neutrosophic),
            # Параметры модели (numpy -> list для чистого pickle)
            'partitions': partitions_data,
            'rule_indices': self.rule_indices.tolist(),
            'consequent_params': self.consequent_params.tolist(),
            # Границы данных
            'mins': self.mins.tolist(),
            'maxs': self.maxs.tolist(),
            # Нейтрософские интервалы
            'neutrosophic_intervals': neutrosophic_intervals,
            # Scaler параметры (если есть)
            'scaler_mean': self.scaler.mean_.tolist() if self.scaler is not None else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.scaler is not None else None,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Модель сохранена в {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'TakagiSugenoClassifier':
        """Загрузка модели из файла (чистый pickle формат)."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        config = ModelConfig(
            n_inputs=save_dict['n_inputs'],
            n_classes=save_dict['n_classes'],
            n_mfs=save_dict['n_mfs'],
            temperature=save_dict.get('temperature', 1.0),
            use_neutrosophic=save_dict.get('use_neutrosophic', False),
        )
        
        model = cls(config=config)
        model._actual_n_inputs = save_dict['n_inputs']
        model.partitions = [FuzzyPartition.from_dict(p) for p in save_dict['partitions']]
        model.rule_indices = np.array(save_dict['rule_indices'], dtype=np.int32)
        model.consequent_params = np.array(save_dict['consequent_params'], dtype=np.float64)
        model.mins = np.array(save_dict['mins'], dtype=np.float64)
        model.maxs = np.array(save_dict['maxs'], dtype=np.float64)
        
        # Восстановление scaler
        if save_dict.get('scaler_mean') is not None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.mean_ = np.array(save_dict['scaler_mean'])
            scaler.scale_ = np.array(save_dict['scaler_scale'])
            scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
            model.scaler = scaler
        
        # Восстановление нейтрософского обработчика
        if config.use_neutrosophic and save_dict.get('neutrosophic_intervals') is not None:
            neutro_config = NeutrosophicConfig(
                intervals=np.array(save_dict['neutrosophic_intervals'])
            )
            model.neutrosophic_handler = NeutrosophicHandler(neutro_config)
            model.firing_strategy = NeutrosophicFiringStrategy(model.neutrosophic_handler)
        else:
            model.firing_strategy = ClassicalFiringStrategy()
        
        model.is_fitted = True
        print(f"Модель загружена из {path}")
        return model


# ============================================================================
# АНСАМБЛЬ КЛАССИФИКАТОРОВ (для повышения точности)
# ============================================================================

class EnsembleTakagiSugeno:
    """
    Ансамбль классификаторов Такаги-Сугено.
    
    Использует несколько моделей с разными параметрами и усредняет
    их предсказания (soft voting) для повышения точности.
    """
    
    def __init__(
        self,
        n_estimators: int = 5,
        base_config: Optional[ModelConfig] = None,
        diversity: bool = True,
        use_neutrosophic: bool = False
    ):
        """
        Args:
            n_estimators: Количество моделей в ансамбле
            base_config: Базовая конфигурация модели
            diversity: Использовать разные параметры для разнообразия
            use_neutrosophic: Использовать нейтрософскую логику
        """
        self.n_estimators = n_estimators
        self.base_config = base_config
        self.diversity = diversity
        self.use_neutrosophic = use_neutrosophic
        self.models: List[TakagiSugenoClassifier] = []
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
    
    def _create_diverse_configs(self) -> List[ModelConfig]:
        """Создать разнообразные конфигурации для моделей ансамбля."""
        configs = []
        base = self.base_config
        
        # Вариации параметров для разнообразия
        n_mfs_variants = [base.n_mfs - 1, base.n_mfs, base.n_mfs + 1, base.n_mfs, base.n_mfs + 2]
        temp_variants = [0.3, 0.5, 0.7, 0.4, 0.6]
        reg_variants = [0.005, 0.01, 0.02, 0.015, 0.008]
        overlap_variants = [1.5, 2.0, 2.5, 1.8, 2.2]
        
        for i in range(self.n_estimators):
            if self.diversity:
                config = ModelConfig(
                    n_inputs=base.n_inputs,
                    n_classes=base.n_classes,
                    n_mfs=max(3, n_mfs_variants[i % len(n_mfs_variants)]),
                    regularization=reg_variants[i % len(reg_variants)],
                    temperature=temp_variants[i % len(temp_variants)],
                    use_neutrosophic=self.use_neutrosophic,
                    dynamic_neutrosophic=base.dynamic_neutrosophic,
                    max_rules=base.max_rules + (i - 2) * 20,  # Вариация ±40
                    overlap_factor=overlap_variants[i % len(overlap_variants)]
                )
            else:
                config = base
            configs.append(config)
        
        return configs
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleTakagiSugeno':
        """
        Обучить ансамбль моделей.
        
        Args:
            X: Признаки (n_samples, n_features)
            y: Метки классов (n_samples,)
        """
        print(f"    Обучение ансамбля из {self.n_estimators} моделей...")
        
        configs = self._create_diverse_configs()
        self.models = []
        
        for i, config in enumerate(configs):
            print(f"      Модель {i+1}/{self.n_estimators}: n_mfs={config.n_mfs}, "
                  f"rules={config.max_rules}, temp={config.temperature:.2f}")
            
            model = TakagiSugenoClassifier(config=config)
            model.scaler = self.scaler
            model.fit(X, y)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Получить усреднённые вероятности от всех моделей.
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Вероятности (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Ансамбль не обучен. Вызовите fit() сначала.")
        
        # Собираем вероятности от всех моделей
        all_probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            all_probas.append(proba)
        
        # Усредняем (soft voting)
        avg_proba = np.mean(all_probas, axis=0)
        return avg_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать классы (soft voting).
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Предсказанные классы (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_model_agreements(self, X: np.ndarray) -> np.ndarray:
        """
        Получить степень согласия моделей (для оценки уверенности).
        
        Returns:
            Доля моделей, согласных с итоговым предсказанием (n_samples,)
        """
        predictions = np.array([model.predict(X) for model in self.models])
        final_pred = self.predict(X)
        
        agreements = np.mean(predictions == final_pred, axis=0)
        return agreements


# ============================================================================
# ИЕРАРХИЧЕСКИЙ КЛАССИФИКАТОР (для повышения точности при многих классах)
# ============================================================================

class HierarchicalTakagiSugeno:
    """
    Иерархический классификатор Такаги-Сугено.
    
    Двухуровневая классификация:
    1. Сначала классифицируем в группы (кластеры похожих классов)
    2. Затем классифицируем внутри каждой группы
    
    Это позволяет лучше разделять перекрывающиеся классы.
    """
    
    def __init__(
        self,
        n_groups: int = 10,
        base_config: Optional[ModelConfig] = None,
        use_neutrosophic: bool = False
    ):
        """
        Args:
            n_groups: Количество групп на первом уровне
            base_config: Базовая конфигурация модели
            use_neutrosophic: Использовать нейтрософскую логику
        """
        self.n_groups = n_groups
        self.base_config = base_config
        self.use_neutrosophic = use_neutrosophic
        
        # Модель первого уровня (группы)
        self.group_model: Optional[TakagiSugenoClassifier] = None
        
        # Модели второго уровня (по одной на группу)
        self.class_models: Dict[int, TakagiSugenoClassifier] = {}
        
        # Маппинг класс -> группа
        self.class_to_group: Dict[int, int] = {}
        self.group_to_classes: Dict[int, List[int]] = {}
        
        self.scaler: Optional[StandardScaler] = None
        self.n_classes: int = 0
        self.is_fitted = False
    
    def _create_groups(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Создать группы классов на основе центроидов.
        Используем K-Means для кластеризации центроидов классов.
        """
        from sklearn.cluster import KMeans
        
        # Вычисляем центроид каждого класса
        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)
        
        centroids = []
        class_list = []
        for c in unique_classes:
            mask = y == c
            centroid = X[mask].mean(axis=0)
            centroids.append(centroid)
            class_list.append(c)
        
        centroids = np.array(centroids)
        
        # Кластеризуем центроиды
        n_groups = min(self.n_groups, len(unique_classes))
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(centroids)
        
        # Создаём маппинг
        self.class_to_group = {class_list[i]: group_labels[i] for i in range(len(class_list))}
        
        self.group_to_classes = {}
        for c, g in self.class_to_group.items():
            if g not in self.group_to_classes:
                self.group_to_classes[g] = []
            self.group_to_classes[g].append(c)
        
        print(f"    Создано {n_groups} групп:")
        for g, classes in sorted(self.group_to_classes.items()):
            print(f"      Группа {g}: {len(classes)} классов")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HierarchicalTakagiSugeno':
        """
        Обучить иерархический классификатор.
        
        Args:
            X: Признаки (n_samples, n_features)
            y: Метки классов (n_samples,)
        """
        print(f"    Создание групп классов...")
        self._create_groups(X, y)
        
        # Создаём метки групп для обучающих данных
        y_groups = np.array([self.class_to_group[c] for c in y])
        
        # 1. Обучаем модель первого уровня (предсказание группы)
        print(f"\n    [Уровень 1] Обучение классификатора групп...")
        n_groups = len(self.group_to_classes)
        
        group_config = ModelConfig(
            n_inputs=self.base_config.n_inputs,
            n_classes=n_groups,
            n_mfs=max(3, self.base_config.n_mfs - 1),  # Меньше ФП для групп
            regularization=self.base_config.regularization,
            temperature=self.base_config.temperature,
            use_neutrosophic=self.use_neutrosophic,
            max_rules=min(150, n_groups * 10),
            overlap_factor=self.base_config.overlap_factor
        )
        
        self.group_model = TakagiSugenoClassifier(config=group_config)
        self.group_model.scaler = self.scaler
        self.group_model.fit(X, y_groups)
        
        # Оценка точности первого уровня
        y_groups_pred = self.group_model.predict(X)
        group_acc = accuracy_score(y_groups, y_groups_pred)
        print(f"    Точность классификации групп (train): {group_acc:.4f}")
        
        # 2. Обучаем модели второго уровня (по одной на группу)
        print(f"\n    [Уровень 2] Обучение классификаторов внутри групп...")
        self.class_models = {}
        
        for group_id, class_list in sorted(self.group_to_classes.items()):
            if len(class_list) == 1:
                # Если в группе один класс - не нужен классификатор
                print(f"      Группа {group_id}: 1 класс (пропуск)")
                continue
            
            # Отбираем данные этой группы
            mask = np.isin(y, class_list)
            X_group = X[mask]
            y_group = y[mask]
            
            # Переиндексируем классы внутри группы (0, 1, 2, ...)
            class_mapping = {c: i for i, c in enumerate(sorted(class_list))}
            y_group_local = np.array([class_mapping[c] for c in y_group])
            
            # Создаём конфигурацию для этой группы
            n_classes_in_group = len(class_list)
            class_config = ModelConfig(
                n_inputs=self.base_config.n_inputs,
                n_classes=n_classes_in_group,
                n_mfs=self.base_config.n_mfs,
                regularization=self.base_config.regularization,
                temperature=self.base_config.temperature,
                use_neutrosophic=self.use_neutrosophic,
                max_rules=min(100, n_classes_in_group * 5),
                overlap_factor=self.base_config.overlap_factor
            )
            
            model = TakagiSugenoClassifier(config=class_config)
            model.scaler = self.scaler
            model.fit(X_group, y_group_local)
            
            # Сохраняем модель и обратный маппинг
            self.class_models[group_id] = {
                'model': model,
                'class_mapping': class_mapping,
                'inv_mapping': {v: k for k, v in class_mapping.items()}
            }
            
            print(f"      Группа {group_id}: {n_classes_in_group} классов, {len(X_group)} образцов")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать классы (двухуровневая классификация).
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Предсказанные классы (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # 1. Предсказываем группу
        group_pred = self.group_model.predict(X)
        
        # 2. Для каждой группы предсказываем класс
        for group_id in np.unique(group_pred):
            mask = group_pred == group_id
            X_group = X[mask]
            
            if group_id in self.class_models:
                # Используем модель группы
                model_info = self.class_models[group_id]
                local_pred = model_info['model'].predict(X_group)
                # Преобразуем обратно в глобальные индексы классов
                global_pred = np.array([model_info['inv_mapping'][p] for p in local_pred])
                predictions[mask] = global_pred
            else:
                # Группа с одним классом
                single_class = self.group_to_classes[group_id][0]
                predictions[mask] = single_class
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Получить вероятности классов.
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Вероятности (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes))
        
        # Получаем вероятности групп
        group_proba = self.group_model.predict_proba(X)
        
        # Для каждой группы
        for group_id, class_list in self.group_to_classes.items():
            group_prob = group_proba[:, group_id:group_id+1]  # (n_samples, 1)
            
            if group_id in self.class_models:
                # Получаем вероятности внутри группы
                model_info = self.class_models[group_id]
                local_proba = model_info['model'].predict_proba(X)
                
                # Распределяем по глобальным индексам
                for local_idx, global_idx in model_info['inv_mapping'].items():
                    proba[:, global_idx] = group_prob.flatten() * local_proba[:, local_idx]
            else:
                # Группа с одним классом
                single_class = class_list[0]
                proba[:, single_class] = group_prob.flatten()
        
        return proba


# ============================================================================
# БУСТИНГ КЛАССИФИКАТОР (АДАПТИВНОЕ ОБУЧЕНИЕ)
# ============================================================================

class BoostedTakagiSugeno:
    """
    Бустинг классификатор Такаги-Сугено.
    
    Использует адаптивное обучение с весами примеров:
    1. Начальные веса равны 1/n
    2. После каждого раунда увеличиваем веса неправильно классифицированных примеров
    3. Следующая модель обучается с учётом этих весов (через передискретизацию)
    4. Финальное предсказание - взвешенное голосование всех моделей
    """
    
    def __init__(
        self,
        n_rounds: int = 5,
        learning_rate: float = 0.3,
        base_config: Optional[ModelConfig] = None,
        use_neutrosophic: bool = False
    ):
        """
        Args:
            n_rounds: Количество раундов бустинга
            learning_rate: Скорость обучения (влияет на обновление весов)
            base_config: Базовая конфигурация модели
            use_neutrosophic: Использовать нейтрософскую логику
        """
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.base_config = base_config
        self.use_neutrosophic = use_neutrosophic
        
        self.models: List[TakagiSugenoClassifier] = []
        self.model_weights: List[float] = []  # Веса моделей
        
        self.scaler: Optional[StandardScaler] = None
        self.n_classes: int = 0
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BoostedTakagiSugeno':
        """
        Обучить бустинг ансамбль.
        
        Args:
            X: Признаки (n_samples, n_features)
            y: Метки классов (n_samples,)
        """
        n_samples = X.shape[0]
        # Используем max(y) + 1 для корректного определения n_classes
        self.n_classes = int(np.max(y)) + 1
        
        # Инициализация весов примеров (равномерные)
        sample_weights = np.ones(n_samples) / n_samples
        
        self.models = []
        self.model_weights = []
        
        print(f"    Бустинг: {self.n_rounds} раундов...")
        
        for round_idx in range(self.n_rounds):
            print(f"\n      Раунд {round_idx + 1}/{self.n_rounds}:")
            
            # Создаём взвешенную выборку через передискретизацию
            # Используем веса как вероятности для сэмплирования с заменой
            indices = np.random.choice(
                n_samples, 
                size=n_samples, 
                replace=True, 
                p=sample_weights
            )
            X_sampled = X[indices]
            y_sampled = y[indices]
            
            # Вариации параметров для разнообразия моделей
            n_mfs_var = self.base_config.n_mfs + round_idx % 3 - 1  # 4, 5, 6, 4, 5...
            n_mfs_var = max(3, min(9, n_mfs_var))
            
            rules_var = self.base_config.max_rules + (round_idx - 2) * 30
            rules_var = max(200, min(500, rules_var))
            
            # Важно: используем self.n_classes (от всех данных), а не от сэмпла
            config = ModelConfig(
                n_inputs=self.base_config.n_inputs,
                n_classes=self.n_classes,  # Используем глобальное количество классов
                n_mfs=n_mfs_var,
                regularization=self.base_config.regularization * (1 + round_idx * 0.1),
                temperature=self.base_config.temperature,
                use_neutrosophic=self.use_neutrosophic,
                max_rules=rules_var,
                overlap_factor=self.base_config.overlap_factor
            )
            
            # Обучаем модель
            model = TakagiSugenoClassifier(config=config)
            model.scaler = self.scaler
            model.fit(X_sampled, y_sampled)
            
            # Оцениваем точность на исходных данных
            y_pred = model.predict(X)
            errors = (y_pred != y).astype(float)
            
            # Взвешенная ошибка
            weighted_error = np.sum(sample_weights * errors)
            
            # Если ошибка слишком большая или слишком маленькая - пропускаем
            if weighted_error >= 0.5 or weighted_error < 1e-10:
                print(f"        Ошибка={weighted_error:.4f} - пропуск")
                continue
            
            # Вес модели
            model_weight = self.learning_rate * np.log((1 - weighted_error) / weighted_error)
            
            # Обновляем веса примеров
            # Увеличиваем веса неправильно классифицированных
            sample_weights *= np.exp(model_weight * errors)
            sample_weights /= np.sum(sample_weights)  # Нормализация
            
            self.models.append(model)
            self.model_weights.append(model_weight)
            
            # Текущая точность ансамбля (используем внутренний метод без проверки is_fitted)
            if len(self.models) > 1:
                y_pred_ensemble = self._predict_internal(X)
                accuracy = accuracy_score(y, y_pred_ensemble)
            else:
                accuracy = 1 - weighted_error
            
            print(f"        n_mfs={n_mfs_var}, rules={rules_var}")
            print(f"        Ошибка={weighted_error:.4f}, Вес модели={model_weight:.4f}")
            print(f"        Текущая точность ансамбля: {accuracy:.4f}")
        
        if len(self.models) == 0:
            # Если все раунды пропущены, обучаем одну базовую модель
            print("      Бустинг не сработал, обучаем базовую модель...")
            model = TakagiSugenoClassifier(config=self.base_config)
            model.scaler = self.scaler
            model.fit(X, y)
            self.models.append(model)
            self.model_weights.append(1.0)
        
        self.is_fitted = True
        print(f"\n    Итого моделей в ансамбле: {len(self.models)}")
        return self
    
    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        """Внутренний метод без проверки is_fitted (для использования в fit)."""
        n_samples = X.shape[0]
        weighted_proba = np.zeros((n_samples, self.n_classes))
        total_weight = sum(self.model_weights)
        
        for model, weight in zip(self.models, self.model_weights):
            proba = model.predict_proba(X)
            weighted_proba += weight * proba
        
        return weighted_proba / total_weight
    
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Внутренний метод предсказания без проверки is_fitted."""
        proba = self._predict_proba_internal(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Получить вероятности через взвешенное голосование.
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Вероятности (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        n_samples = X.shape[0]
        weighted_proba = np.zeros((n_samples, self.n_classes))
        total_weight = sum(self.model_weights)
        
        for model, weight in zip(self.models, self.model_weights):
            proba = model.predict_proba(X)
            weighted_proba += weight * proba
        
        return weighted_proba / total_weight
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказать классы (взвешенное голосование).
        
        Args:
            X: Признаки (n_samples, n_features)
            
        Returns:
            Предсказанные классы (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ============================================================================
# КЛАСС ВИЗУАЛИЗАЦИИ (SOLID - Единственная ответственность)
# ============================================================================

class ModelVisualizer:
    """
    Инструменты визуализации для результатов классификатора Такаги-Сугено.
    
    Предоставляет комплексные возможности построения графиков:
    - Матрица ошибок
    - ROC-кривые (многоклассовые)
    - Кривые точности-полноты
    - Распределение классов
    - Распределение уверенности
    - Результаты кросс-валидации
    - Сравнение моделей
    - Визуализация функций принадлежности
    """
    
    def __init__(
        self,
        class_names: List[str],
        figsize: Tuple[int, int] = (10, 8),
        save_dpi: int = 150
    ):
        """
        Инициализация визуализатора.
        
        Аргументы:
            class_names: Список названий классов для подписей
            figsize: Размер рисунка по умолчанию
            save_dpi: DPI для сохранённых рисунков
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        # Адаптивный размер для большого числа классов
        if self.n_classes > 20:
            self.figsize = (max(14, self.n_classes // 5), max(10, self.n_classes // 6))
        else:
            self.figsize = figsize
        self.save_dpi = save_dpi
        self.colors = sns.color_palette("husl", len(class_names))
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Матрица ошибок",
        normalize: bool = True,
        save_path: Optional[str] = None,
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Построение матрицы ошибок с аннотациями."""
        cm = confusion_matrix(y_true, y_pred)
        n_classes_actual = cm.shape[0]
        
        if normalize:
            cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm_display = cm
        
        # Адаптивный размер для большого числа классов
        if n_classes_actual > 20:
            fig_size = (max(16, n_classes_actual // 4), max(14, n_classes_actual // 4))
            annot = False  # Отключаем аннотации для большого числа классов
            fontsize = 4
        else:
            fig_size = self.figsize
            annot = True
            fontsize = 10
        
        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            fig = ax.figure
        
        # Для большого числа классов используем только цвета без аннотаций
        sns.heatmap(
            cm_display,
            annot=annot,
            fmt='.0%' if normalize else 'd',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Доля' if normalize else 'Количество'},
            annot_kws={'size': fontsize}
        )
        
        ax.set_xlabel('Предсказанный класс', fontsize=12)
        ax.set_ylabel('Истинный класс', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Подписи осей только для небольшого числа классов
        if n_classes_actual <= 30:
            ax.set_xticklabels(range(n_classes_actual), rotation=90, fontsize=6)
            ax.set_yticklabels(range(n_classes_actual), rotation=0, fontsize=6)
        else:
            # Показываем только каждый 10-й класс
            ax.set_xticks(np.arange(0, n_classes_actual, 10))
            ax.set_xticklabels(np.arange(0, n_classes_actual, 10), rotation=0, fontsize=8)
            ax.set_yticks(np.arange(0, n_classes_actual, 10))
            ax.set_yticklabels(np.arange(0, n_classes_actual, 10), rotation=0, fontsize=8)
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC-кривые",
        save_path: Optional[str] = None,
        top_n: int = 10  # Показать топ-N классов по AUC
    ) -> plt.Figure:
        """Построение многоклассовых ROC-кривых."""
        n_classes = y_proba.shape[1]
        y_true_bin = one_hot_encode(y_true, n_classes)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Вычисляем AUC для всех классов
        roc_auc_dict = {}
        fpr_dict, tpr_dict = {}, {}
        
        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        # Микро-среднее
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        # Для большого числа классов показываем только топ-N по AUC + микро-среднее
        if n_classes > 15:
            # Сортируем по AUC и берём топ-N
            sorted_classes = sorted(roc_auc_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            colors = sns.color_palette("husl", top_n + 1)
            for idx, (i, auc_val) in enumerate(sorted_classes):
                ax.plot(fpr_dict[i], tpr_dict[i], color=colors[idx], lw=2,
                       label=f'Класс_{i} (AUC = {auc_val:.3f})')
            
            ax.plot(fpr_micro, tpr_micro, color='navy', linestyle='--', lw=3,
                   label=f'Микро-среднее (AUC = {roc_auc_micro:.3f})')
            
            title = f"{title} (Топ-{top_n} из {n_classes} классов по AUC)"
        else:
            colors = cycle(self.colors)
            for i, color in zip(range(n_classes), colors):
                class_name = self.class_names[i] if i < len(self.class_names) else f'Класс_{i}'
                ax.plot(fpr_dict[i], tpr_dict[i], color=color, lw=2,
                       label=f'{class_name} (AUC = {roc_auc_dict[i]:.3f})')
            
            ax.plot(fpr_micro, tpr_micro, color='navy', linestyle='--', lw=3,
                   label=f'Микро-среднее (AUC = {roc_auc_micro:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Частота ложноположительных (FPR)', fontsize=12)
        ax.set_ylabel('Частота истинноположительных (TPR)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Кривые точности-полноты",
        save_path: Optional[str] = None,
        top_n: int = 10
    ) -> plt.Figure:
        """Построение многоклассовых кривых точности-полноты."""
        n_classes = y_proba.shape[1]
        y_true_bin = one_hot_encode(y_true, n_classes)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Вычисляем AP для всех классов
        ap_dict = {}
        pr_curves = {}
        
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            ap_dict[i] = np.mean(precision)
            pr_curves[i] = (precision, recall)
        
        # Для большого числа классов показываем только топ-N по AP
        if n_classes > 15:
            sorted_classes = sorted(ap_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            colors = sns.color_palette("husl", top_n)
            for idx, (i, ap_val) in enumerate(sorted_classes):
                precision, recall = pr_curves[i]
                ax.plot(recall, precision, color=colors[idx], lw=2,
                       label=f'Класс_{i} (AP = {ap_val:.3f})')
            
            title = f"{title} (Топ-{top_n} из {n_classes} классов по AP)"
        else:
            colors = cycle(self.colors)
            for i, color in zip(range(n_classes), colors):
                precision, recall = pr_curves[i]
                class_name = self.class_names[i] if i < len(self.class_names) else f'Класс_{i}'
                ax.plot(recall, precision, color=color, lw=2,
                       label=f'{class_name} (AP = {ap_dict[i]:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Полнота (Recall)', fontsize=12)
        ax.set_ylabel('Точность (Precision)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        title: str = "Распределение классов",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение распределения классов в обучающей и тестовой выборках."""
        n_classes = max(int(np.max(y_train)), int(np.max(y_test))) + 1
        train_counts = np.bincount(y_train, minlength=n_classes)
        test_counts = np.bincount(y_test, minlength=n_classes)
        
        # Для большого числа классов используем сводную статистику
        if n_classes > 30:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Гистограмма распределения размеров классов
            axes[0, 0].hist(train_counts, bins=20, alpha=0.7, color='steelblue', label='Обучающая')
            axes[0, 0].hist(test_counts, bins=20, alpha=0.7, color='coral', label='Тестовая')
            axes[0, 0].set_xlabel('Количество примеров в классе')
            axes[0, 0].set_ylabel('Количество классов')
            axes[0, 0].set_title('Распределение размеров классов', fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Топ-20 крупнейших классов
            top_indices = np.argsort(train_counts)[-20:][::-1]
            x = np.arange(20)
            width = 0.35
            axes[0, 1].bar(x - width/2, train_counts[top_indices], width, label='Обучающая', color='steelblue')
            axes[0, 1].bar(x + width/2, test_counts[top_indices], width, label='Тестовая', color='coral')
            axes[0, 1].set_xlabel('Класс')
            axes[0, 1].set_ylabel('Количество')
            axes[0, 1].set_title('Топ-20 крупнейших классов', fontweight='bold')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels([f'{i}' for i in top_indices], rotation=45, fontsize=8)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # 3. Статистика
            stats_text = f"""Статистика распределения классов:
            
Всего классов: {n_classes}
Обучающая выборка: {len(y_train)} примеров
Тестовая выборка: {len(y_test)} примеров

Обучающая:
  Мин: {train_counts.min()}, Макс: {train_counts.max()}
  Среднее: {train_counts.mean():.1f}, Медиана: {np.median(train_counts):.1f}
  
Тестовая:
  Мин: {test_counts.min()}, Макс: {test_counts.max()}
  Среднее: {test_counts.mean():.1f}, Медиана: {np.median(test_counts):.1f}
  
Пустые классы (train): {np.sum(train_counts == 0)}
Пустые классы (test): {np.sum(test_counts == 0)}"""
            
            axes[1, 0].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                           verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
            axes[1, 0].set_title('Сводная статистика', fontweight='bold')
            
            # 4. Boxplot сравнение
            axes[1, 1].boxplot([train_counts, test_counts], labels=['Обучающая', 'Тестовая'])
            axes[1, 1].set_ylabel('Количество примеров в классе')
            axes[1, 1].set_title('Сравнение распределений', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            fig.suptitle(f'{title} ({n_classes} классов)', fontsize=14, fontweight='bold')
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            x = np.arange(n_classes)
            width = 0.35
            axes[0].bar(x, train_counts, color='steelblue')
            axes[0].set_title('Обучающая выборка', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Класс', fontsize=11)
            axes[0].set_ylabel('Количество', fontsize=11)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([f'{i}' for i in range(n_classes)], rotation=45, fontsize=8)
            
            axes[1].bar(x, test_counts, color='coral')
            axes[1].set_title('Тестовая выборка', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Класс', fontsize=11)
            axes[1].set_ylabel('Количество', fontsize=11)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([f'{i}' for i in range(n_classes)], rotation=45, fontsize=8)
            
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_confidence_distribution(
        self,
        confidences: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Распределение уверенности предсказаний",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение распределения уверенности предсказаний."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        correct_mask = (y_true == y_pred)
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]
        
        axes[0].hist(correct_conf, bins=20, alpha=0.7, label=f'Правильные ({len(correct_conf)})',
                    color='green', edgecolor='darkgreen')
        if len(incorrect_conf) > 0:
            axes[0].hist(incorrect_conf, bins=20, alpha=0.7, label=f'Ошибочные ({len(incorrect_conf)})',
                        color='red', edgecolor='darkred')
        
        axes[0].set_xlabel('Уверенность', fontsize=11)
        axes[0].set_ylabel('Количество', fontsize=11)
        axes[0].set_title('Уверенность по правильности предсказания', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        n_classes = int(np.max(y_true)) + 1
        
        # Для большого числа классов показываем сводную статистику
        if n_classes > 30:
            # Вычисляем среднюю уверенность по классам
            mean_conf_by_class = [np.mean(confidences[y_true == i]) if np.any(y_true == i) else 0 for i in range(n_classes)]
            
            # Гистограмма средних уверенностей
            axes[1].hist(mean_conf_by_class, bins=20, color='steelblue', edgecolor='darkblue', alpha=0.7)
            axes[1].axvline(np.mean(mean_conf_by_class), color='red', linestyle='--', lw=2, 
                           label=f'Среднее: {np.mean(mean_conf_by_class):.3f}')
            axes[1].set_xlabel('Средняя уверенность по классу', fontsize=11)
            axes[1].set_ylabel('Количество классов', fontsize=11)
            axes[1].set_title(f'Распределение средней уверенности ({n_classes} классов)', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            conf_by_class = [confidences[y_true == i] if np.any(y_true == i) else np.array([0]) for i in range(n_classes)]
            class_labels = [f'{i}' for i in range(n_classes)]
            bp = axes[1].boxplot(conf_by_class, tick_labels=class_labels, patch_artist=True)
            colors_extended = self.colors * (n_classes // len(self.colors) + 1)
            for patch, color in zip(bp['boxes'], colors_extended):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            axes[1].set_xlabel('Класс', fontsize=11)
            axes[1].set_ylabel('Уверенность', fontsize=11)
            axes[1].set_title('Распределение уверенности по классам', fontsize=12, fontweight='bold')
            axes[1].tick_params(axis='x', rotation=45, labelsize=8)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_cross_validation_results(
        self,
        cv_results: Dict[str, any],
        title: str = "Результаты кросс-валидации",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение результатов кросс-валидации."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        fold_accuracies = cv_results['fold_accuracies']
        n_folds = len(fold_accuracies)
        folds = list(range(1, n_folds + 1))
        
        bars = axes[0].bar(folds, fold_accuracies, color=self.colors[:n_folds], edgecolor='black')
        axes[0].axhline(y=cv_results['accuracy_mean'], color='red', linestyle='--', 
                       linewidth=2, label=f"Среднее: {cv_results['accuracy_mean']:.4f}")
        axes[0].fill_between(
            [0.5, n_folds + 0.5],
            cv_results['accuracy_mean'] - cv_results['accuracy_std'],
            cv_results['accuracy_mean'] + cv_results['accuracy_std'],
            alpha=0.2, color='red', label=f"±1 СКО: {cv_results['accuracy_std']:.4f}"
        )
        
        for bar, acc in zip(bars, fold_accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        axes[0].set_xlabel('Фолд', fontsize=11)
        axes[0].set_ylabel('Точность', fontsize=11)
        axes[0].set_title('Точность по фолдам', fontsize=12, fontweight='bold')
        axes[0].set_ylim([min(fold_accuracies) - 0.05, 1.02])
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        metrics = {
            'Точность': (cv_results['accuracy_mean'], cv_results['accuracy_std']),
            'F1-мера': (cv_results['f1_mean'], cv_results['f1_std'])
        }
        
        metric_names = list(metrics.keys())
        means = [m[0] for m in metrics.values()]
        stds = [m[1] for m in metrics.values()]
        
        x_pos = np.arange(len(metric_names))
        bars2 = axes[1].bar(x_pos, means, yerr=stds, capsize=5, 
                           color=['steelblue', 'coral'], edgecolor='black')
        
        for bar, mean, std in zip(bars2, means, stds):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=10)
        
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(metric_names, fontsize=11)
        axes[1].set_ylabel('Значение', fontsize=11)
        axes[1].set_title('Сводные метрики', fontsize=12, fontweight='bold')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        title: str = "Сравнение моделей",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Сравнение нескольких моделей по различным метрикам."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        model_names = list(model_results.keys())
        metrics = list(next(iter(model_results.values())).keys())
        
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        # Контрастные цвета для сравнения моделей
        comparison_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
        
        for i, (model_name, results) in enumerate(model_results.items()):
            values = [results[m] for m in metrics]
            offset = (i - len(model_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name, 
                         color=comparison_colors[i % len(comparison_colors)])
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylabel('Значение', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.15])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_architecture_info(
        self,
        hyper_config: 'HyperConfig',
        model_config: 'ModelConfig',
        data_info: Dict[str, any],
        training_results: Dict[str, float],
        title: str = "Архитектура и гиперпараметры модели",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализация архитектуры модели и гиперпараметров.
        
        Args:
            hyper_config: Конфигурация гиперпараметров
            model_config: Конфигурация модели
            data_info: Информация о данных {'n_samples', 'n_features', 'n_classes', 'train_size', 'test_size'}
            training_results: Результаты обучения {'accuracy', 'f1', 'cv_accuracy', 'cv_std'}
            title: Заголовок графика
            save_path: Путь для сохранения
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # ========== 1. АРХИТЕКТУРА (верхний левый) ==========
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        # Определяем тип архитектуры
        if hyper_config.use_boosting:
            arch_type = "БУСТИНГ"
            arch_color = '#e74c3c'
            arch_details = [
                f"Раундов: {hyper_config.n_boosting_rounds}",
                f"Скорость обучения: {hyper_config.boosting_learning_rate}",
                "Адаптивное взвешивание"
            ]
        elif hyper_config.use_hierarchical:
            arch_type = "ИЕРАРХИЧЕСКАЯ"
            arch_color = '#9b59b6'
            arch_details = [
                f"Групп: {hyper_config.n_groups}",
                "Двухуровневая классификация",
                "K-Means кластеризация"
            ]
        elif hyper_config.use_ensemble:
            arch_type = "АНСАМБЛЬ"
            arch_color = '#3498db'
            arch_details = [
                f"Моделей: {hyper_config.n_estimators}",
                f"Разнообразие: {'Да' if hyper_config.ensemble_diversity else 'Нет'}",
                "Голосование большинством"
            ]
        else:
            arch_type = "ОДИНОЧНАЯ"
            arch_color = '#2ecc71'
            arch_details = [
                "Базовый Такаги-Сугено",
                "Без ансамблирования",
                "Прямой вывод"
            ]
        
        # Нейтрософская логика
        neutro_status = "Нейтрософская" if model_config.use_neutrosophic else "Классическая"
        
        # Рисуем блок архитектуры как список
        ax1.text(0.5, 0.75, arch_type, transform=ax1.transAxes, fontsize=20,
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=arch_color, alpha=0.3))
        
        details_text = f"{arch_details[0]}\n{arch_details[1]}\n{arch_details[2]}\n\nЛогика: {neutro_status}"
        ax1.text(0.5, 0.35, details_text, transform=ax1.transAxes, fontsize=12,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        ax1.set_title('Архитектура', fontsize=14, fontweight='bold', pad=10)
        
        # ========== 2. ГИПЕРПАРАМЕТРЫ (верхний правый) ==========
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        params = [
            ("Функции принадлежности", f"{model_config.n_mfs}"),
            ("Макс. правил", f"{model_config.max_rules}"),
            ("Регуляризация", f"{model_config.regularization}"),
            ("Температура", f"{model_config.temperature}"),
            ("Перекрытие ФП", f"{model_config.overlap_factor}"),
            ("PCA", f"{'Да (' + str(int(hyper_config.pca_variance*100)) + '%)' if hyper_config.use_pca else 'Нет'}"),
            ("Тест. выборка", f"{int(hyper_config.test_size*100)}%"),
            ("Random seed", f"{hyper_config.random_state}"),
        ]
        
        table_data = [[p[0], p[1]] for p in params]
        table = ax2.table(cellText=table_data, colLabels=['Параметр', 'Значение'],
                         loc='center', cellLoc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Цвета для таблицы
        for i in range(len(params) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Заголовок
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(color='white', fontweight='bold')
                else:
                    cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
        
        ax2.set_title('Гиперпараметры', fontsize=14, fontweight='bold', pad=10)
        
        # ========== 3. ДАННЫЕ (нижний левый) ==========
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Круговая диаграмма train/test
        train_pct = 100 - int(hyper_config.test_size * 100)
        test_pct = int(hyper_config.test_size * 100)
        sizes = [train_pct, test_pct]
        labels = [f'Обучение\n{data_info.get("train_size", "N/A")} примеров\n({train_pct}%)',
                  f'Тест\n{data_info.get("test_size", "N/A")} примеров\n({test_pct}%)']
        colors_pie = ['#2ecc71', '#e74c3c']
        explode = (0.02, 0.02)
        
        wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                           autopct='', startangle=90, 
                                           wedgeprops=dict(width=0.5, edgecolor='white'))
        
        # Центральный текст
        centre_text = f"Всего\n{data_info.get('n_samples', 'N/A')}\nпримеров"
        ax3.text(0, 0, centre_text, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Информация о данных справа
        data_text = f"""
Признаков: {data_info.get('n_features', 'N/A')}
Классов: {data_info.get('n_classes', 'N/A')}
Правил: {len(model_config.max_rules) if hasattr(model_config.max_rules, '__len__') else model_config.max_rules}
"""
        ax3.text(1.3, 0, data_text.strip(), ha='left', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
        
        ax3.set_title('Данные', fontsize=14, fontweight='bold', pad=10)
        
        # ========== 4. РЕЗУЛЬТАТЫ (нижний правый) ==========
        ax4 = fig.add_subplot(gs[1, 1])
        
        metrics_names = ['Точность\n(Тест)', 'F1-мера\n(Тест)', 'CV Точность', 'CV СКО']
        metrics_values = [
            training_results.get('accuracy', 0),
            training_results.get('f1', 0),
            training_results.get('cv_accuracy', 0),
            training_results.get('cv_std', 0)
        ]
        
        # Цвета в зависимости от значения
        bar_colors = []
        for i, v in enumerate(metrics_values):
            if i == 3:  # CV std - меньше лучше
                bar_colors.append('#2ecc71' if v < 0.05 else '#f39c12' if v < 0.1 else '#e74c3c')
            else:
                bar_colors.append('#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.6 else '#e74c3c')
        
        x_pos = np.arange(len(metrics_names))
        bars = ax4.bar(x_pos, metrics_values, color=bar_colors, edgecolor='black', alpha=0.8)
        
        # Значения над столбцами
        for bar, val in zip(bars, metrics_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(metrics_names, fontsize=10)
        ax4.set_ylim([0, 1.15])
        ax4.set_ylabel('Значение', fontsize=11)
        ax4.set_title('Результаты обучения', fontsize=14, fontweight='bold', pad=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Легенда цветов
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Отлично (>=80%)'),
            Patch(facecolor='#f39c12', label='Средне (60-80%)'),
            Patch(facecolor='#e74c3c', label='Низко (<60%)')
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig

    def plot_membership_functions(
        self,
        partition: 'FuzzyPartition',
        input_name: str = "Вход",
        n_points: int = 200,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение функций принадлежности для нечёткого разбиения."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.linspace(partition.min_val, partition.max_val, n_points)
        mf_labels = ["Очень низкий", "Низкий", "Средний", "Высокий", "Очень высокий", "Очень высокий+", "Макс"]
        
        for i in range(partition.n_mfs):
            memberships = gaussian_membership_vectorized(
                x, 
                np.array([partition.centers[i]]),
                np.array([partition.sigmas[i]])
            ).flatten()
            
            label = mf_labels[i] if i < len(mf_labels) else f"ФП{i}"
            ax.plot(x, memberships, label=label, linewidth=2, 
                   color=self.colors[i % len(self.colors)])
            ax.fill_between(x, memberships, alpha=0.2, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel(input_name, fontsize=12)
        ax.set_ylabel('Степень принадлежности', fontsize=12)
        ax.set_title(f'Функции принадлежности для {input_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_metrics_summary(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Модель",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение сводки метрик на одном рисунке."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        n_unique = len(unique_labels)
        
        # 1. Матрица ошибок
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_confusion_matrix(y_true, y_pred, title="Матрица ошибок", ax=ax1)
        
        # 2. Метрики - для большого числа классов показываем распределение
        ax2 = fig.add_subplot(gs[0, 1])
        precision = precision_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, labels=unique_labels, average=None, zero_division=0)
        
        if n_unique > 20:
            # Гистограммы распределения метрик
            ax2.hist(precision, bins=15, alpha=0.6, label=f'Precision (ср.={np.mean(precision):.3f})', color='steelblue')
            ax2.hist(recall, bins=15, alpha=0.6, label=f'Recall (ср.={np.mean(recall):.3f})', color='coral')
            ax2.hist(f1, bins=15, alpha=0.6, label=f'F1 (ср.={np.mean(f1):.3f})', color='green')
            ax2.set_xlabel('Значение метрики')
            ax2.set_ylabel('Количество классов')
            ax2.set_title(f'Распределение метрик ({n_unique} классов)', fontweight='bold')
            ax2.legend(fontsize=8)
        else:
            x = np.arange(n_unique)
            width = 0.25
            ax2.bar(x - width, precision, width, label='Precision', color='steelblue')
            ax2.bar(x, recall, width, label='Recall', color='coral')
            ax2.bar(x + width, f1, width, label='F1', color='green')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{i}' for i in unique_labels], rotation=45, fontsize=8)
            ax2.set_ylabel('Значение')
            ax2.set_title('Метрики по классам', fontweight='bold')
            ax2.legend(loc='lower right', fontsize=8)
            ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Гистограмма уверенности
        ax3 = fig.add_subplot(gs[0, 2])
        confidences = np.max(y_proba, axis=1)
        correct_mask = (y_true == y_pred)
        ax3.hist(confidences[correct_mask], bins=15, alpha=0.7, label='Правильные', color='green')
        if np.any(~correct_mask):
            ax3.hist(confidences[~correct_mask], bins=15, alpha=0.7, label='Ошибочные', color='red')
        ax3.set_xlabel('Уверенность')
        ax3.set_ylabel('Количество')
        ax3.set_title('Распределение уверенности', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. ROC-кривая
        ax4 = fig.add_subplot(gs[1, 0])
        n_classes = y_proba.shape[1]  # Используем размерность из y_proba
        y_true_bin = one_hot_encode(y_true, n_classes)
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax4.plot(fpr_micro, tpr_micro, color='navy', lw=2,
                label=f'ROC (AUC = {roc_auc_micro:.3f})')
        ax4.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1.05])
        ax4.set_xlabel('FPR')
        ax4.set_ylabel('TPR')
        ax4.set_title('ROC-кривая (микро-среднее)', fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Общие метрики
        ax5 = fig.add_subplot(gs[1, 1])
        overall_metrics = {
            'Точность': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1-мера': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        bars = ax5.bar(overall_metrics.keys(), overall_metrics.values(), 
                      color=['steelblue', 'coral', 'green', 'purple'])
        for bar, val in zip(bars, overall_metrics.values()):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Значение')
        ax5.set_title('Общие метрики', fontweight='bold')
        ax5.set_ylim([0, 1.15])
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Круговая диаграмма предсказаний
        ax6 = fig.add_subplot(gs[1, 2])
        correct_count = np.sum(correct_mask)
        incorrect_count = len(correct_mask) - correct_count
        ax6.pie([correct_count, incorrect_count], 
               labels=[f'Правильные\n({correct_count})', f'Ошибочные\n({incorrect_count})'],
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90, explode=(0.05, 0.05))
        ax6.set_title('Точность предсказаний', fontweight='bold')
        
        fig.suptitle(f'{model_name} - Сводка производительности', fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def plot_neutrosophic_analysis(
        self,
        confidences: Dict[str, np.ndarray],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Нейтрософский анализ неопределённости",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Построение специфического для нейтрософской логики анализа."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        correct_mask = (y_true == y_pred)
        
        # 1. Вероятность vs Неопределённость
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            confidences['probability'], confidences['indeterminacy'],
            c=correct_mask.astype(int), cmap='RdYlGn', alpha=0.7,
            edgecolors='black', linewidths=0.5
        )
        ax1.set_xlabel('Вероятность', fontsize=11)
        ax1.set_ylabel('Неопределённость', fontsize=11)
        ax1.set_title('Вероятность vs Неопределённость', fontsize=12, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Правильно (1) / Ошибочно (0)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Гистограмма нейтрософской уверенности
        ax2 = axes[0, 1]
        ax2.hist(confidences['neutrosophic_confidence'][correct_mask], bins=15, 
                alpha=0.7, label='Правильные', color='green', edgecolor='darkgreen')
        if np.any(~correct_mask):
            ax2.hist(confidences['neutrosophic_confidence'][~correct_mask], bins=15, 
                    alpha=0.7, label='Ошибочные', color='red', edgecolor='darkred')
        ax2.set_xlabel('Нейтрософская уверенность', fontsize=11)
        ax2.set_ylabel('Количество', fontsize=11)
        ax2.set_title('Распределение нейтрософской уверенности', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Сравнение компонентов
        ax3 = axes[1, 0]
        components = ['Вероятность', 'Неопределённость', 'Нейтро. увер.']
        correct_means = [
            np.mean(confidences['probability'][correct_mask]),
            np.mean(confidences['indeterminacy'][correct_mask]),
            np.mean(confidences['neutrosophic_confidence'][correct_mask])
        ]
        incorrect_means = [
            np.mean(confidences['probability'][~correct_mask]) if np.any(~correct_mask) else 0,
            np.mean(confidences['indeterminacy'][~correct_mask]) if np.any(~correct_mask) else 0,
            np.mean(confidences['neutrosophic_confidence'][~correct_mask]) if np.any(~correct_mask) else 0
        ]
        
        x = np.arange(len(components))
        width = 0.35
        ax3.bar(x - width/2, correct_means, width, label='Правильные', color='green')
        ax3.bar(x + width/2, incorrect_means, width, label='Ошибочные', color='red')
        ax3.set_xticks(x)
        ax3.set_xticklabels(components)
        ax3.set_ylabel('Среднее значение', fontsize=11)
        ax3.set_title('Средние компоненты: Правильные vs Ошибочные', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Боксплоты/гистограмма по классам
        ax4 = axes[1, 1]
        n_classes = int(np.max(y_true)) + 1
        
        if n_classes > 30:
            # Для большого числа классов - гистограмма средних
            mean_neutro_by_class = [np.mean(confidences['neutrosophic_confidence'][y_true == i]) 
                                   if np.any(y_true == i) else 0 for i in range(n_classes)]
            ax4.hist(mean_neutro_by_class, bins=20, color='purple', alpha=0.7, edgecolor='darkviolet')
            ax4.axvline(np.mean(mean_neutro_by_class), color='red', linestyle='--', lw=2,
                       label=f'Среднее: {np.mean(mean_neutro_by_class):.3f}')
            ax4.set_xlabel('Средняя нейтрософская уверенность', fontsize=11)
            ax4.set_ylabel('Количество классов', fontsize=11)
            ax4.set_title(f'Распределение по {n_classes} классам', fontsize=12, fontweight='bold')
            ax4.legend()
        else:
            neutro_conf_by_class = [confidences['neutrosophic_confidence'][y_true == i] if np.any(y_true == i) else np.array([0])
                                   for i in range(n_classes)]
            class_labels = [f'{i}' for i in range(n_classes)]
            bp = ax4.boxplot(neutro_conf_by_class, tick_labels=class_labels, patch_artist=True)
            colors_extended = self.colors * (n_classes // len(self.colors) + 1)
            for patch, color in zip(bp['boxes'], colors_extended):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax4.set_xlabel('Класс', fontsize=11)
            ax4.set_ylabel('Нейтрософская уверенность', fontsize=11)
            ax4.set_title('Нейтрософская уверенность по классам', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.save_dpi, bbox_inches='tight')
            print(f"Сохранено: {save_path}")
        
        return fig
    
    def create_full_report(
        self,
        model: 'TakagiSugenoClassifier',
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        cv_results: Optional[Dict] = None,
        output_dir: str = "plots",
        model_name: str = "ТакагиСугено"
    ) -> Dict[str, plt.Figure]:
        """Создание полного отчёта с визуализациями и сохранение всех графиков."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        figures['metrics_summary'] = self.plot_metrics_summary(
            y_test, y_pred, y_proba, model_name,
            save_path=f"{output_dir}/{model_name}_metrics_summary.png"
        )
        
        figures['confusion_matrix'] = self.plot_confusion_matrix(
            y_test, y_pred, title=f"{model_name} - Матрица ошибок",
            save_path=f"{output_dir}/{model_name}_confusion_matrix.png"
        )
        
        figures['roc_curves'] = self.plot_roc_curves(
            y_test, y_proba, title=f"{model_name} - ROC-кривые",
            save_path=f"{output_dir}/{model_name}_roc_curves.png"
        )
        
        figures['pr_curves'] = self.plot_precision_recall_curves(
            y_test, y_proba, title=f"{model_name} - Кривые точности-полноты",
            save_path=f"{output_dir}/{model_name}_pr_curves.png"
        )
        
        figures['class_dist'] = self.plot_class_distribution(
            y_train, y_test, title=f"{model_name} - Распределение классов",
            save_path=f"{output_dir}/{model_name}_class_distribution.png"
        )
        
        confidences = np.max(y_proba, axis=1)
        figures['confidence_dist'] = self.plot_confidence_distribution(
            confidences, y_test, y_pred,
            title=f"{model_name} - Распределение уверенности",
            save_path=f"{output_dir}/{model_name}_confidence_distribution.png"
        )
        
        if cv_results:
            figures['cv_results'] = self.plot_cross_validation_results(
                cv_results, title=f"{model_name} - Cross-Validation Results",
                save_path=f"{output_dir}/{model_name}_cv_results.png"
            )
        
        if model.partitions:
            figures['membership_funcs'] = self.plot_membership_functions(
                model.partitions[0], input_name="Feature 0",
                save_path=f"{output_dir}/{model_name}_membership_functions.png"
            )
        
        if isinstance(model.firing_strategy, NeutrosophicFiringStrategy):
            conf_info = model.get_prediction_confidence(X_test)
            figures['neutrosophic_analysis'] = self.plot_neutrosophic_analysis(
                conf_info, y_test, y_pred,
                title=f"{model_name} - Neutrosophic Analysis",
                save_path=f"{output_dir}/{model_name}_neutrosophic_analysis.png"
            )
        
        print(f"\nВсе графики сохранены в папку '{output_dir}/'")
        return figures


# ============================================================================
# ЗАГРУЗКА ДАННЫХ (DRY - Одна функция)
# ============================================================================

@dataclass
class DataResult:
    """Результат загрузки данных."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    n_classes: int
    n_features: int
    class_names: Dict[str, int]
    class_names_inv: Dict[int, str]
    pca: Optional[Any] = None           # PCA трансформер (если использовался)
    n_features_original: int = 0        # Исходное количество признаков до PCA


def load_data(
    filepath: str = None,
    hyper_config: Optional[HyperConfig] = None,
    config_module = None
) -> DataResult:
    """
    Загрузка и предобработка данных с опциональным PCA.
    
    Автоматически определяет:
    - Количество классов из данных
    - Имена классов (из config или автоматически)
    - Количество признаков
    
    Args:
        filepath: Путь к файлу данных (или из config)
        hyper_config: Конфигурация гиперпараметров
        config_module: Модуль конфигурации (config.py)
    """
    from sklearn.decomposition import PCA
    
    # Загружаем config если не передан
    if config_module is None:
        try:
            import config as config_module
        except ImportError:
            config_module = None
    
    if hyper_config is None:
        hyper_config = HYPER_CONFIG
    
    # Получаем параметры из config или используем значения по умолчанию
    if config_module:
        filepath = filepath or getattr(config_module, 'DATA_FILE', 'train2.csv')
        separator = getattr(config_module, 'DATA_SEPARATOR', ';')
        decimal = getattr(config_module, 'DATA_DECIMAL', ',')
        has_header = getattr(config_module, 'HAS_HEADER', False)
        class_start_index = getattr(config_module, 'CLASS_START_INDEX', 1)
        class_names_config = getattr(config_module, 'CLASS_NAMES', None)
    else:
        filepath = filepath or 'train2.csv'
        separator = ';'
        decimal = ','
        has_header = False
        class_start_index = 1
        class_names_config = None
    
    # Получаем параметры из конфигурации
    test_size = hyper_config.test_size
    random_state = hyper_config.random_state
    
    # Загрузка данных (поддержка .csv и .txt)
    header = 0 if has_header else None
    
    # Автоопределение кодировки
    encoding = None
    try:
        with open(filepath, 'rb') as f:
            first_bytes = f.read(4)
            if first_bytes[:2] == b'\xff\xfe':
                encoding = 'utf-16-le'
                print(f"    📝 Обнаружена кодировка: UTF-16 LE")
            elif first_bytes[:2] == b'\xfe\xff':
                encoding = 'utf-16-be'
                print(f"    📝 Обнаружена кодировка: UTF-16 BE")
            elif first_bytes[:3] == b'\xef\xbb\xbf':
                encoding = 'utf-8-sig'
                print(f"    📝 Обнаружена кодировка: UTF-8 BOM")
    except:
        pass
    
    # Определяем расширение файла
    file_ext = filepath.lower().split('.')[-1] if '.' in filepath else ''
    
    if file_ext in ['csv', 'txt', 'data', 'dat']:
        # Читаем как текстовый файл с разделителями
        data = pd.read_csv(
            filepath, 
            sep=separator, 
            decimal=decimal, 
            header=header,
            engine='python',  # Более гибкий парсер
            encoding=encoding
        )
    else:
        # Пробуем автоопределение формата
        data = pd.read_csv(filepath, sep=separator, decimal=decimal, header=header, encoding=encoding)
    
    # === СОХРАНЯЕМ ОРИГИНАЛЬНЫЙ СТОЛБЕЦ КЛАССОВ (до конвертации) ===
    y_original = data.iloc[:, -1].copy()
    is_text_classes = not pd.api.types.is_numeric_dtype(y_original)
    
    if is_text_classes:
        print(f"    📝 Обнаружены текстовые названия классов")
        # Создаём маппинг текст -> число
        unique_labels = sorted(y_original.unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        print(f"    📋 Найдено {len(unique_labels)} уникальных классов")
    
    # Конвертируем в числовой формат (только признаки, без последнего столбца)
    data_features = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    
    # === АНАЛИЗ КАЧЕСТВА ДАННЫХ (только для признаков) ===
    nan_per_column = data_features.isna().sum()
    total_nan = nan_per_column.sum()
    total_cells = data_features.shape[0] * data_features.shape[1]
    nan_percentage = total_nan / total_cells * 100 if total_cells > 0 else 0
    
    if total_nan > 0:
        print(f"\n    ⚠️  ОБНАРУЖЕНЫ ПРОБЛЕМЫ С ДАННЫМИ!")
        print(f"    📊 Всего ячеек: {total_cells}")
        print(f"    ❌ Некорректных ячеек: {total_nan} ({nan_percentage:.1f}%)")
        print(f"    📋 Строк с проблемами: {data_features.isna().any(axis=1).sum()}")
        
        # Детали по столбцам
        problem_cols = nan_per_column[nan_per_column > 0]
        if len(problem_cols) > 0:
            print(f"\n    Проблемные столбцы:")
            for col in problem_cols.index:
                col_nan = problem_cols[col]
                col_pct = col_nan / len(data) * 100
                severity = "🔴" if col_pct > 50 else ("🟡" if col_pct > 10 else "🟢")
                print(f"      {severity} Столбец {col}: {col_nan} NaN ({col_pct:.1f}%)")
        
        # Порог критичности
        if nan_percentage > 20:
            print(f"\n    🚨 КРИТИЧНО: {nan_percentage:.1f}% данных повреждены!")
            print(f"    Рекомендация: Исправьте исходный файл данных.")
            print(f"    Возможные причины:")
            print(f"      - Неправильная кодировка файла")
            print(f"      - Смешанные десятичные разделители (точка/запятая)")
            print(f"      - Повреждение при экспорте из Excel")
            raise ValueError(f"Слишком много повреждённых данных: {nan_percentage:.1f}% (порог: 20%)")
        elif nan_percentage > 5:
            print(f"\n    ⚠️  ВНИМАНИЕ: {nan_percentage:.1f}% данных повреждены!")
            print(f"    Продолжаю с удалением проблемных строк...")
            # Находим индексы строк без NaN
            valid_rows = ~data_features.isna().any(axis=1)
            data_features = data_features[valid_rows]
            y_original = y_original[valid_rows]
            print(f"    Удалено строк: {(~valid_rows).sum()}")
        else:
            print(f"\n    ℹ️  Незначительные проблемы ({nan_percentage:.1f}%), удаляю проблемные строки...")
            valid_rows = ~data_features.isna().any(axis=1)
            data_features = data_features[valid_rows]
            y_original = y_original[valid_rows]
    else:
        print(f"    ✅ Данные корректны, проблем не обнаружено")
    
    # Извлечение X
    X = data_features.values
    
    # Обработка классов
    if is_text_classes:
        # Текстовые классы -> числа
        y = np.array([label_to_idx[label] for label in y_original])
        n_classes = len(unique_labels)
        
        # Создаём маппинг имён
        class_names = {label: idx for label, idx in label_to_idx.items()}
        class_names_inv = idx_to_label
        
        print(f"    ✅ Текстовые классы конвертированы: {n_classes} классов")
    else:
        # Числовые классы
        y_raw = y_original.values
        y = y_raw.astype(int) - class_start_index
    
        # Определяем уникальные классы и их количество
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        # Сохраняем оригинальные номера классов (до переиндексации)
        original_class_labels = [c + class_start_index for c in sorted(unique_classes)]  # Возвращаем к оригинальным номерам
        
        # Создаём маппинг если классы не последовательные
        if not np.array_equal(unique_classes, np.arange(n_classes)):
            class_mapping = {old: new for new, old in enumerate(sorted(unique_classes))}
            # Обратный маппинг: внутренний индекс -> оригинальный номер (с учётом class_start_index)
            reverse_mapping = {new: old + class_start_index for old, new in class_mapping.items()}
            y = np.array([class_mapping[c] for c in y])
            print(f"    ⚠️  Классы переиндексированы: {len(unique_classes)} уникальных классов")
        else:
            reverse_mapping = {i: i + class_start_index for i in range(n_classes)}
        
        # Генерируем или загружаем имена классов
        class_names = {}
        class_names_inv = {}
        
        if class_names_config is None:
            # Автоматическое именование с ОРИГИНАЛЬНЫМИ номерами классов
            for i in range(n_classes):
                original_label = reverse_mapping.get(i, i + class_start_index)
                name = f"Класс_{original_label}"
                class_names[name] = i
                class_names_inv[i] = name
        elif isinstance(class_names_config, dict):
            # Словарь {индекс: имя}
            for idx, name in class_names_config.items():
                class_names[name] = idx
                class_names_inv[idx] = name
            # Добавляем недостающие классы
            for i in range(n_classes):
                if i not in class_names_inv:
                    name = f"Класс_{i}"
                    class_names[name] = i
                    class_names_inv[i] = name
        elif isinstance(class_names_config, list):
            # Список имён
            for i, name in enumerate(class_names_config):
                class_names[name] = i
                class_names_inv[i] = name
            # Добавляем недостающие классы
            for i in range(len(class_names_config), n_classes):
                name = f"Класс_{i}"
                class_names[name] = i
                class_names_inv[i] = name
        elif isinstance(class_names_config, str):
            # Путь к файлу с именами классов
            try:
                with open(class_names_config, 'r', encoding='utf-8') as f:
                    names = [line.strip() for line in f if line.strip()]
                for i, name in enumerate(names):
                    class_names[name] = i
                    class_names_inv[i] = name
                # Добавляем недостающие классы
                for i in range(len(names), n_classes):
                    name = f"Класс_{i}"
                    class_names[name] = i
                    class_names_inv[i] = name
            except FileNotFoundError:
                print(f"    ⚠️  Файл {class_names_config} не найден, используем автоимена")
                for i in range(n_classes):
                    name = f"Класс_{i}"
                    class_names[name] = i
                    class_names_inv[i] = name
    
    n_features_original = X.shape[1]
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Стандартизация
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Применение PCA (если включено)
    pca = None
    if hyper_config.use_pca:
        if hyper_config.pca_n_components is not None:
            n_components = hyper_config.pca_n_components
        else:
            n_components = hyper_config.pca_variance
        
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        print(f"    PCA: {n_features_original} → {X_train.shape[1]} признаков "
              f"({pca.explained_variance_ratio_.sum():.1%} дисперсии)")
    
    n_features = X_train.shape[1]
    
    return DataResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        n_classes=n_classes,
        n_features=n_features,
        class_names=class_names,
        class_names_inv=class_names_inv,
        pca=pca,
        n_features_original=n_features_original
    )


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main(config_module=None):
    """
    Главная точка входа.
    
    Args:
        config_module: Модуль конфигурации (config.py). Если None, загружается автоматически.
    """
    # Загружаем config если не передан
    if config_module is None:
        try:
            import config as config_module
        except ImportError:
            config_module = None
    
    print("=" * 70)
    print("ОПТИМИЗИРОВАННЫЙ нечёткий классификатор Такаги-Сугено")
    print("(Векторизация NumPy/SciPy, SOLID, DRY)")
    print("=" * 70)
    
    # Загрузка данных (параметры берутся из config автоматически)
    print("\n[1] Загрузка данных...")
    data_result = load_data(hyper_config=HYPER_CONFIG, config_module=config_module)
    print(f"    Обучающая: {len(data_result.X_train)}, Тестовая: {len(data_result.X_test)}, "
          f"Признаки: {data_result.n_features}, Классы: {data_result.n_classes}")
    
    # Получение адаптивных параметров из централизованной конфигурации
    hyper_params = HYPER_CONFIG.calculate_adaptive_params(
        n_classes=data_result.n_classes,
        n_features=data_result.n_features
    )
    
    print(f"\n    Гиперпараметры (из HYPER_CONFIG):")
    print(f"      n_mfs={hyper_params['n_mfs']}, max_rules={hyper_params['max_rules']}")
    print(f"      regularization={hyper_params['regularization']}, temperature={hyper_params['temperature']}")
    print(f"      overlap_factor={hyper_params['overlap_factor']}")
    print(f"      use_boosting={HYPER_CONFIG.use_boosting}, n_rounds={HYPER_CONFIG.n_boosting_rounds}")
    
    config = ModelConfig(
        n_inputs=data_result.n_features,
        n_classes=data_result.n_classes,
        n_mfs=hyper_params['n_mfs'],
        regularization=hyper_params['regularization'],
        temperature=hyper_params['temperature'],
        use_neutrosophic=False,
        max_rules=hyper_params['max_rules'],
        overlap_factor=hyper_params['overlap_factor']
    )
    
    # ========== БУСТИНГ, ИЕРАРХИЧЕСКАЯ ИЛИ ОБЫЧНАЯ МОДЕЛЬ ==========
    if HYPER_CONFIG.use_boosting:
        print("\n[2] Обучение БУСТИНГ классификатора...")
        boosted = BoostedTakagiSugeno(
            n_rounds=HYPER_CONFIG.n_boosting_rounds,
            learning_rate=HYPER_CONFIG.boosting_learning_rate,
            base_config=config,
            use_neutrosophic=False
        )
        boosted.scaler = data_result.scaler
        boosted.fit(data_result.X_train, data_result.y_train)
        
        print("\n[3] Оценка бустинг классификатора...")
        y_pred = boosted.predict(data_result.X_test)
        accuracy = accuracy_score(data_result.y_test, y_pred)
        print(f"    Точность бустинга на тесте: {accuracy:.4f}")
        
        # Для сравнения - обычная модель
        print("\n[3.1] Для сравнения - обычная (одиночная) модель...")
        single_model = TakagiSugenoClassifier(config=config)
        single_model.scaler = data_result.scaler
        single_model.fit(data_result.X_train, data_result.y_train)
        y_pred_single = single_model.predict(data_result.X_test)
        accuracy_single = accuracy_score(data_result.y_test, y_pred_single)
        print(f"    Точность одиночной модели: {accuracy_single:.4f}")
        print(f"    Улучшение от бустинга: {(accuracy - accuracy_single)*100:+.2f}%")
        
        model = single_model  # Для совместимости с остальным кодом
    elif HYPER_CONFIG.use_hierarchical:
        print("\n[2] Обучение ИЕРАРХИЧЕСКОГО классификатора...")
        hierarchical = HierarchicalTakagiSugeno(
            n_groups=HYPER_CONFIG.n_groups,
            base_config=config,
            use_neutrosophic=False
        )
        hierarchical.scaler = data_result.scaler
        hierarchical.fit(data_result.X_train, data_result.y_train)
        
        print("\n[3] Оценка иерархического классификатора...")
        y_pred = hierarchical.predict(data_result.X_test)
        accuracy = accuracy_score(data_result.y_test, y_pred)
        print(f"    Точность иерархического на тесте: {accuracy:.4f}")
        
        # Для сравнения - обычная модель
        print("\n[3.1] Для сравнения - обычная (плоская) модель...")
        single_model = TakagiSugenoClassifier(config=config)
        single_model.scaler = data_result.scaler
        single_model.fit(data_result.X_train, data_result.y_train)
        y_pred_single = single_model.predict(data_result.X_test)
        accuracy_single = accuracy_score(data_result.y_test, y_pred_single)
        print(f"    Точность обычной модели: {accuracy_single:.4f}")
        print(f"    Улучшение от иерархии: {(accuracy - accuracy_single)*100:+.2f}%")
        
        model = single_model  # Для совместимости с остальным кодом
    elif HYPER_CONFIG.use_ensemble:
        print("\n[2] Обучение АНСАМБЛЯ классических моделей...")
        ensemble = EnsembleTakagiSugeno(
            n_estimators=HYPER_CONFIG.n_estimators,
            base_config=config,
            diversity=HYPER_CONFIG.ensemble_diversity,
            use_neutrosophic=False
        )
        ensemble.scaler = data_result.scaler
        ensemble.fit(data_result.X_train, data_result.y_train)
        
        print("\n[3] Оценка ансамбля...")
        y_pred = ensemble.predict(data_result.X_test)
        accuracy = accuracy_score(data_result.y_test, y_pred)
        print(f"    Точность ансамбля на тесте: {accuracy:.4f}")
        
        # Согласие моделей
        agreements = ensemble.get_model_agreements(data_result.X_test)
        print(f"    Среднее согласие моделей: {agreements.mean():.2%}")
        
        # Для сравнения - одиночная модель
        print("\n[3.1] Для сравнения - одиночная модель...")
        single_model = TakagiSugenoClassifier(config=config)
        single_model.scaler = data_result.scaler
        single_model.fit(data_result.X_train, data_result.y_train)
        y_pred_single = single_model.predict(data_result.X_test)
        accuracy_single = accuracy_score(data_result.y_test, y_pred_single)
        print(f"    Точность одиночной модели: {accuracy_single:.4f}")
        print(f"    Улучшение от ансамбля: +{(accuracy - accuracy_single)*100:.2f}%")
        
        model = single_model  # Для совместимости с остальным кодом
    else:
        # Классическая одиночная модель
        print("\n[2] Обучение классической модели...")
        model = TakagiSugenoClassifier(config=config)
        model.scaler = data_result.scaler
        model.fit(data_result.X_train, data_result.y_train)
        
        print("\n[3] Оценка...")
        y_pred = model.predict(data_result.X_test)
        accuracy = accuracy_score(data_result.y_test, y_pred)
        print(f"    Точность на тесте: {accuracy:.4f}")
    
    print("\n    Отчёт классификации:")
    unique_labels = np.unique(np.concatenate([data_result.y_test, y_pred]))
    target_names = [data_result.class_names_inv.get(i, f"Класс_{i}") for i in unique_labels]
    print(classification_report(data_result.y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
    
    # ========== АНАЛИЗ КАЧЕСТВА КЛАССОВ ==========
    print("\n" + "=" * 70)
    print("АНАЛИЗ КАЧЕСТВА КЛАССОВ")
    print("=" * 70)
    
    # Получаем F1-score для каждого класса
    from sklearn.metrics import f1_score as f1_score_func
    
    # Вычисляем F1 для каждого класса
    unique_classes = np.unique(data_result.y_test)
    class_f1_scores = {}
    
    for cls in unique_classes:
        # Создаём бинарную маску для текущего класса
        y_true_binary = (data_result.y_test == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)
        
        # Вычисляем F1-score для этого класса
        f1 = f1_score_func(y_true_binary, y_pred_binary, zero_division=0)
        class_name = data_result.class_names_inv.get(cls, f"Класс_{cls}")
        class_f1_scores[class_name] = f1 * 100  # В процентах
    
    # Группируем классы по качеству
    excellent_classes = []  # 80-100%
    good_classes = []       # 70-80%
    poor_classes = []       # < 70%
    
    for class_name, f1 in class_f1_scores.items():
        if f1 >= 80:
            excellent_classes.append((class_name, f1))
        elif f1 >= 70:
            good_classes.append((class_name, f1))
        else:
            poor_classes.append((class_name, f1))
    
    # Сортируем по F1-score (убывание)
    excellent_classes.sort(key=lambda x: x[1], reverse=True)
    good_classes.sort(key=lambda x: x[1], reverse=True)
    poor_classes.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🟢 ОТЛИЧНЫЕ КЛАССЫ (F1 ≥ 80%): {len(excellent_classes)} классов")
    print("-" * 50)
    for i, (name, f1) in enumerate(excellent_classes):
        print(f"    {i+1:3d}. {name:15s} : {f1:5.1f}%")
    
    print(f"\n🟡 СРЕДНИЕ КЛАССЫ (70% ≤ F1 < 80%): {len(good_classes)} классов")
    print("-" * 50)
    for i, (name, f1) in enumerate(good_classes):
        print(f"    {i+1:3d}. {name:15s} : {f1:5.1f}%")
    
    print(f"\n🔴 ПЛОХИЕ КЛАССЫ (F1 < 70%): {len(poor_classes)} классов")
    print("-" * 50)
    for i, (name, f1) in enumerate(poor_classes):
        print(f"    {i+1:3d}. {name:15s} : {f1:5.1f}%")
    
    # Статистика
    print(f"\n" + "=" * 50)
    print("СТАТИСТИКА:")
    print(f"    Всего классов:      {len(class_f1_scores)}")
    print(f"    Отличных (≥80%):    {len(excellent_classes)} ({len(excellent_classes)/len(class_f1_scores)*100:.1f}%)")
    print(f"    Средних (70-80%):   {len(good_classes)} ({len(good_classes)/len(class_f1_scores)*100:.1f}%)")
    print(f"    Плохих (<70%):      {len(poor_classes)} ({len(poor_classes)/len(class_f1_scores)*100:.1f}%)")
    
    if excellent_classes:
        avg_excellent = np.mean([f1 for _, f1 in excellent_classes])
        print(f"    Средний F1 отличных: {avg_excellent:.1f}%")
    if good_classes:
        avg_good = np.mean([f1 for _, f1 in good_classes])
        print(f"    Средний F1 средних:  {avg_good:.1f}%")
    if poor_classes:
        avg_poor = np.mean([f1 for _, f1 in poor_classes])
        print(f"    Средний F1 плохих:   {avg_poor:.1f}%")
    print("=" * 50)
    
    # Кросс-валидация (только для одиночной модели)
    print("\n[4] Кросс-валидация...")
    
    # Получаем параметры из config
    if config_module:
        data_file = getattr(config_module, 'DATA_FILE', 'train2.csv')
        separator = getattr(config_module, 'DATA_SEPARATOR', ';')
        decimal = getattr(config_module, 'DATA_DECIMAL', ',')
        has_header = getattr(config_module, 'HAS_HEADER', False)
        class_start_index = getattr(config_module, 'CLASS_START_INDEX', 1)
        cv_folds = getattr(config_module, 'CV_FOLDS', 5)
    else:
        cv_folds = 5
    
    # Используем уже загруженные и переиндексированные данные
    X_full = np.vstack([data_result.X_train, data_result.X_test])
    y_full = np.concatenate([data_result.y_train, data_result.y_test])
    
    cv_results = model.cross_validate(X_full, y_full, n_folds=cv_folds)
    print(f"    Точность CV: {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
    
    # ========== НЕЙТРОСОФСКИЙ АНСАМБЛЬ ==========
    print("\n" + "=" * 70)
    print("НЕЙТРОСОФСКАЯ МОДЕЛЬ" + (" (АНСАМБЛЬ)" if HYPER_CONFIG.use_ensemble else ""))
    print("=" * 70)
    
    neutro_config = ModelConfig(
        n_inputs=data_result.n_features,
        n_classes=data_result.n_classes,
        n_mfs=hyper_params['n_mfs'],
        regularization=hyper_params['regularization'],
        temperature=hyper_params['temperature'],
        use_neutrosophic=True,
        dynamic_neutrosophic=True,
        max_rules=hyper_params['max_rules'],
        overlap_factor=hyper_params['overlap_factor']
    )
    
    if HYPER_CONFIG.use_ensemble:
        print("\n[5] Обучение АНСАМБЛЯ нейтрософских моделей...")
        neutro_ensemble = EnsembleTakagiSugeno(
            n_estimators=HYPER_CONFIG.n_estimators,
            base_config=neutro_config,
            diversity=HYPER_CONFIG.ensemble_diversity,
            use_neutrosophic=True
        )
        neutro_ensemble.scaler = data_result.scaler
        neutro_ensemble.fit(data_result.X_train, data_result.y_train)
        
        print("\n[6] Оценка нейтрософского ансамбля...")
        y_pred_neutro = neutro_ensemble.predict(data_result.X_test)
        accuracy_neutro = accuracy_score(data_result.y_test, y_pred_neutro)
        print(f"    Точность нейтрософского ансамбля: {accuracy_neutro:.4f}")
        
        neutro_model = TakagiSugenoClassifier(config=neutro_config)  # Для совместимости
        neutro_model.scaler = data_result.scaler
        neutro_model.fit(data_result.X_train, data_result.y_train)
    else:
        print("\n[5] Обучение нейтрософской модели...")
        neutro_model = TakagiSugenoClassifier(config=neutro_config)
        neutro_model.scaler = data_result.scaler
        neutro_model.fit(data_result.X_train, data_result.y_train)
        
        print("\n[6] Оценка нейтрософской модели...")
        y_pred_neutro = neutro_model.predict(data_result.X_test)
        accuracy_neutro = accuracy_score(data_result.y_test, y_pred_neutro)
        print(f"    Точность на тесте: {accuracy_neutro:.4f}")
    
    # Предсказания с неопределённостью
    print("\n[7] Предсказания с неопределённостью...")
    for i in range(min(3, len(data_result.X_test))):
        pred, proba, indet = neutro_model.predict_with_uncertainty(data_result.X_test[i:i+1])
        conf = neutro_model.get_prediction_confidence(data_result.X_test[i:i+1])
        
        true_label = data_result.class_names_inv[data_result.y_test[i]]
        pred_label = data_result.class_names_inv[pred[0]]
        status = "✓" if pred[0] == data_result.y_test[i] else "✗"
        
        print(f"    Пример {i}: Истинный={true_label}, Предсказанный={pred_label}")
        print(f"      Нейтрософская уверенность: {conf['neutrosophic_confidence'][0]:.2%} {status}")
    
    # Тест сохранения/загрузки
    print("\n[8] Тест сохранения/загрузки...")
    model.save_model("ts_optimized_classical.pkl")
    neutro_model.save_model("ts_optimized_neutro.pkl")
    
    loaded = TakagiSugenoClassifier.load_model("ts_optimized_classical.pkl")
    loaded_pred = loaded.predict(data_result.X_test[:5])
    print(f"    Предсказания совпадают: {np.array_equal(y_pred[:5], loaded_pred)}")
    
    # Сравнение
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ")
    print("=" * 70)
    print(f"    Точность классической:     {accuracy:.4f}")
    print(f"    Точность нейтрософской:    {accuracy_neutro:.4f}")
    
    diff = np.where(y_pred != y_pred_neutro)[0]
    if len(diff) > 0:
        print(f"    Модели расходятся на {len(diff)} примерах")
    else:
        print("    Модели согласны по всем предсказаниям!")
    
    # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================
    print("\n" + "=" * 70)
    print("ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИЙ")
    print("=" * 70)
    
    visualizer = ModelVisualizer(class_names=list(data_result.class_names.keys()))
    
    # Визуализация классической модели
    print("\n[9] Создание визуализаций классической модели...")
    visualizer.create_full_report(
        model=model,
        X_train=data_result.X_train,
        X_test=data_result.X_test,
        y_train=data_result.y_train,
        y_test=data_result.y_test,
        cv_results=cv_results,
        output_dir="plots_classical",
        model_name="Классический_ТС"
    )
    
    # Визуализация нейтрософской модели
    print("\n[10] Создание визуализаций нейтрософской модели...")
    cv_results_neutro = neutro_model.cross_validate(X_full, y_full, n_folds=5)
    visualizer.create_full_report(
        model=neutro_model,
        X_train=data_result.X_train,
        X_test=data_result.X_test,
        y_train=data_result.y_train,
        y_test=data_result.y_test,
        cv_results=cv_results_neutro,
        output_dir="plots_neutrosophic",
        model_name="Нейтрософский_ТС"
    )
    
    # Сравнение моделей
    print("\n[11] Создание сравнения моделей...")
    y_proba = model.predict_proba(data_result.X_test)
    y_proba_neutro = neutro_model.predict_proba(data_result.X_test)
    
    model_comparison = {
        'Классический ТС': {
            'Точность': accuracy_score(data_result.y_test, y_pred),
            'Прецизионность': precision_score(data_result.y_test, y_pred, average='weighted', zero_division=0),
            'Полнота': recall_score(data_result.y_test, y_pred, average='weighted', zero_division=0),
            'F1-мера': f1_score(data_result.y_test, y_pred, average='weighted', zero_division=0)
        },
        'Нейтрософский ТС': {
            'Точность': accuracy_score(data_result.y_test, y_pred_neutro),
            'Прецизионность': precision_score(data_result.y_test, y_pred_neutro, average='weighted', zero_division=0),
            'Полнота': recall_score(data_result.y_test, y_pred_neutro, average='weighted', zero_division=0),
            'F1-мера': f1_score(data_result.y_test, y_pred_neutro, average='weighted', zero_division=0)
        }
    }
    
    visualizer.plot_model_comparison(
        model_comparison,
        title="Классический vs Нейтрософский Такаги-Сугено",
        save_path="plots_comparison/model_comparison.png"
    )
    
    # Создание матриц ошибок рядом
    import os
    os.makedirs("plots_comparison", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    visualizer.plot_confusion_matrix(
        data_result.y_test, y_pred, title="Классический ТС - Матрица ошибок", ax=axes[0]
    )
    visualizer.plot_confusion_matrix(
        data_result.y_test, y_pred_neutro, title="Нейтрософский ТС - Матрица ошибок", ax=axes[1]
    )
    fig.suptitle("Сравнение матриц ошибок", fontsize=16, fontweight='bold')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    fig.savefig("plots_comparison/confusion_matrices_comparison.png", dpi=150, bbox_inches='tight')
    print("Сохранено: plots_comparison/confusion_matrices_comparison.png")
    
    # Сравнение ROC-кривых рядом (микро-среднее для многоклассовой задачи)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    n_classes = data_result.n_classes
    y_test_bin = one_hot_encode(data_result.y_test, n_classes)
    
    # Классический ROC (микро-среднее)
    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    axes[0].plot(fpr_micro, tpr_micro, lw=2, color='navy', label=f'Микро-среднее (AUC={roc_auc_micro:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_xlabel('Доля ложноположительных (FPR)')
    axes[0].set_ylabel('Доля истинноположительных (TPR)')
    axes[0].set_title('Классический ТС - ROC-кривая', fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Нейтрософский ROC (микро-среднее)
    fpr_micro_n, tpr_micro_n, _ = roc_curve(y_test_bin.ravel(), y_proba_neutro.ravel())
    roc_auc_micro_n = auc(fpr_micro_n, tpr_micro_n)
    axes[1].plot(fpr_micro_n, tpr_micro_n, lw=2, color='darkred', label=f'Микро-среднее (AUC={roc_auc_micro_n:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_xlabel('Доля ложноположительных (FPR)')
    axes[1].set_ylabel('Доля истинноположительных (TPR)')
    axes[1].set_title('Нейтрософский ТС - ROC-кривая', fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Сравнение ROC-кривых", fontsize=16, fontweight='bold')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    fig.savefig("plots_comparison/roc_curves_comparison.png", dpi=150, bbox_inches='tight')
    print("Сохранено: plots_comparison/roc_curves_comparison.png")
    
    # Сравнение распределений уверенности
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Классическая уверенность
    conf_classical = np.max(y_proba, axis=1)
    correct_classical = (y_pred == data_result.y_test)
    axes[0].hist(conf_classical[correct_classical], bins=15, alpha=0.7, 
                label=f'Правильные ({np.sum(correct_classical)})', color='green')
    axes[0].hist(conf_classical[~correct_classical], bins=15, alpha=0.7, 
                label=f'Ошибочные ({np.sum(~correct_classical)})', color='red')
    axes[0].set_xlabel('Уверенность')
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Классический ТС - Распределение уверенности', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Нейтрософская уверенность
    conf_neutro = np.max(y_proba_neutro, axis=1)
    correct_neutro = (y_pred_neutro == data_result.y_test)
    axes[1].hist(conf_neutro[correct_neutro], bins=15, alpha=0.7, 
                label=f'Правильные ({np.sum(correct_neutro)})', color='green')
    if np.sum(~correct_neutro) > 0:
        axes[1].hist(conf_neutro[~correct_neutro], bins=15, alpha=0.7, 
                    label=f'Ошибочные ({np.sum(~correct_neutro)})', color='red')
    axes[1].set_xlabel('Уверенность')
    axes[1].set_ylabel('Количество')
    axes[1].set_title('Нейтрософский ТС - Распределение уверенности', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle("Сравнение распределений уверенности", fontsize=16, fontweight='bold')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    fig.savefig("plots_comparison/confidence_comparison.png", dpi=150, bbox_inches='tight')
    print("Сохранено: plots_comparison/confidence_comparison.png")
    
    # Средняя уверенность - общая статистика (для большого числа классов)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_conf_classical = [np.mean(conf_classical[data_result.y_test == i]) if np.any(data_result.y_test == i) else 0 for i in range(n_classes)]
    mean_conf_neutro = [np.mean(conf_neutro[data_result.y_test == i]) if np.any(data_result.y_test == i) else 0 for i in range(n_classes)]
    
    # Показываем распределение средних уверенностей
    ax.hist(mean_conf_classical, bins=20, alpha=0.7, label=f'Классический ТС (ср.={np.mean(mean_conf_classical):.3f})', color='steelblue')
    ax.hist(mean_conf_neutro, bins=20, alpha=0.7, label=f'Нейтрософский ТС (ср.={np.mean(mean_conf_neutro):.3f})', color='coral')
    
    ax.set_xlabel('Средняя уверенность по классу')
    ax.set_ylabel('Количество классов')
    ax.set_title(f'Распределение средней уверенности по {n_classes} классам', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    fig.savefig("plots_comparison/mean_confidence_per_class.png", dpi=150, bbox_inches='tight')
    print("Сохранено: plots_comparison/mean_confidence_per_class.png")
    
    # ========== ГРАФИК АРХИТЕКТУРЫ И ГИПЕРПАРАМЕТРОВ ==========
    print("\n[12] Создание графика архитектуры и гиперпараметров...")
    
    # Информация о данных
    data_info = {
        'n_samples': len(data_result.X_train) + len(data_result.X_test),
        'n_features': data_result.n_features,
        'n_classes': data_result.n_classes,
        'train_size': len(data_result.X_train),
        'test_size': len(data_result.X_test)
    }
    
    # Результаты обучения для классической модели
    training_results_classical = {
        'accuracy': accuracy,
        'f1': f1_score(data_result.y_test, y_pred, average='weighted', zero_division=0),
        'cv_accuracy': cv_results['accuracy_mean'],
        'cv_std': cv_results['accuracy_std']
    }
    
    # Результаты для нейтрософской модели
    training_results_neutro = {
        'accuracy': accuracy_neutro,
        'f1': f1_score(data_result.y_test, y_pred_neutro, average='weighted', zero_division=0),
        'cv_accuracy': cv_results_neutro['accuracy_mean'],
        'cv_std': cv_results_neutro['accuracy_std']
    }
    
    # График для классической модели
    visualizer.plot_architecture_info(
        hyper_config=HYPER_CONFIG,
        model_config=config,
        data_info=data_info,
        training_results=training_results_classical,
        title="Архитектура: Классический Такаги-Сугено",
        save_path="plots_classical/architecture_info.png"
    )
    
    # График для нейтрософской модели
    visualizer.plot_architecture_info(
        hyper_config=HYPER_CONFIG,
        model_config=neutro_config,
        data_info=data_info,
        training_results=training_results_neutro,
        title="Архитектура: Нейтрософский Такаги-Сугено",
        save_path="plots_neutrosophic/architecture_info.png"
    )
    
    # Общий сравнительный график архитектуры
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # 1. Тип архитектуры
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    if HYPER_CONFIG.use_boosting:
        arch_name = "БУСТИНГ"
        arch_params = f"Раундов: {HYPER_CONFIG.n_boosting_rounds}\nСкорость: {HYPER_CONFIG.boosting_learning_rate}"
    elif HYPER_CONFIG.use_hierarchical:
        arch_name = "ИЕРАРХИЧЕСКАЯ"
        arch_params = f"Групп: {HYPER_CONFIG.n_groups}\nДвухуровневая"
    elif HYPER_CONFIG.use_ensemble:
        arch_name = "АНСАМБЛЬ"
        arch_params = f"Моделей: {HYPER_CONFIG.n_estimators}\nРазнообразие: {HYPER_CONFIG.ensemble_diversity}"
    else:
        arch_name = "ОДИНОЧНАЯ"
        arch_params = "Базовая модель\nТакаги-Сугено"
    
    ax1.text(0.5, 0.6, f"{arch_name}", transform=ax1.transAxes,
             fontsize=24, ha='center', va='center', fontweight='bold')
    ax1.text(0.5, 0.25, arch_params, transform=ax1.transAxes,
             fontsize=12, ha='center', va='center')
    ax1.set_title('Архитектура', fontsize=14, fontweight='bold')
    
    # 2. Сравнение точности
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Классический\nТС', 'Нейтрософский\nТС']
    accuracies = [accuracy, accuracy_neutro]
    colors_bar = ['#2ecc71', '#e74c3c']
    bars = ax2.bar(models, accuracies, color=colors_bar, edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.1])
    ax2.set_ylabel('Точность')
    ax2.set_title('Сравнение точности', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Параметры модели
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    params_text = f"""
    Функций принадл.: {config.n_mfs}
    Макс. правил: {config.max_rules}
    Регуляризация: {config.regularization}
    Температура: {config.temperature}
    PCA: {'Да (' + str(int(HYPER_CONFIG.pca_variance*100)) + '%)' if HYPER_CONFIG.use_pca else 'Нет'}
    """
    ax3.text(0.5, 0.5, params_text.strip(), transform=ax3.transAxes,
             fontsize=11, ha='center', va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Гиперпараметры', fontsize=14, fontweight='bold')
    
    # 4. Данные
    ax4 = fig.add_subplot(gs[1, 0])
    sizes = [len(data_result.X_train), len(data_result.X_test)]
    labels = [f'Обучение\n{sizes[0]}', f'Тест\n{sizes[1]}']
    colors_pie = ['#3498db', '#e74c3c']
    ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=(0.02, 0.02))
    ax4.text(0, 0, f'{sum(sizes)}\nвсего', ha='center', va='center', fontsize=10, fontweight='bold')
    ax4.set_title(f'Данные ({data_result.n_features} признаков, {data_result.n_classes} классов)', 
                  fontsize=14, fontweight='bold')
    
    # 5. CV результаты
    ax5 = fig.add_subplot(gs[1, 1])
    cv_models = ['Классич.\nТС', 'Нейтрософ.\nТС']
    cv_means = [cv_results['accuracy_mean'], cv_results_neutro['accuracy_mean']]
    cv_stds = [cv_results['accuracy_std'], cv_results_neutro['accuracy_std']]
    x_pos = np.arange(len(cv_models))
    bars = ax5.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, color=colors_bar, edgecolor='black')
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}+-{std:.3f}', ha='center', va='bottom', fontsize=10)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(cv_models)
    ax5.set_ylim([0, 1.15])
    ax5.set_ylabel('CV Точность')
    ax5.set_title('Кросс-валидация', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Различия в предсказаниях
    ax6 = fig.add_subplot(gs[1, 2])
    n_total = len(data_result.y_test)
    n_diff = len(diff)
    n_agree = n_total - n_diff
    sizes_pred = [n_agree, n_diff]
    labels_pred = [f'Совпадают\n{n_agree} ({n_agree/n_total*100:.1f}%)', 
                   f'Расходятся\n{n_diff} ({n_diff/n_total*100:.1f}%)']
    colors_pred = ['#2ecc71', '#f39c12']
    ax6.pie(sizes_pred, labels=labels_pred, colors=colors_pred, 
            startangle=90, explode=(0, 0.05))
    ax6.set_title('Согласованность моделей', fontsize=14, fontweight='bold')
    
    fig.suptitle(f'Обзор архитектуры: {arch_name} Такаги-Сугено', 
                 fontsize=16, fontweight='bold')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    fig.savefig("plots_comparison/architecture_overview.png", dpi=150, bbox_inches='tight')
    print("Сохранено: plots_comparison/architecture_overview.png")
    
    plt.close('all')
    
    print("\n" + "=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 70)
    print("\nСозданные папки с графиками:")
    print("  - plots_classical/       : Визуализации классического Такаги-Сугено")
    print("  - plots_neutrosophic/    : Визуализации нейтрософского Такаги-Сугено")
    print("  - plots_comparison/      : Сравнительные визуализации")
    print("\nФайлы графиков:")
    print("  - *_metrics_summary.png         : Общий обзор метрик")
    print("  - *_confusion_matrix.png        : Тепловая карта матрицы ошибок")
    print("  - *_roc_curves.png              : ROC-кривые (многоклассовые)")
    print("  - *_pr_curves.png               : Кривые точность-полнота")
    print("  - *_class_distribution.png      : Распределение классов в выборках")
    print("  - *_confidence_distribution.png : Анализ уверенности предсказаний")
    print("  - *_cv_results.png              : Результаты кросс-валидации")
    print("  - *_membership_functions.png    : Нечёткие функции принадлежности")
    print("  - *_neutrosophic_analysis.png   : Нейтрософский анализ неопределённости")
    print("  - architecture_info.png         : Архитектура и гиперпараметры модели")
    print("\nСравнительные графики:")
    print("  - model_comparison.png          : Гистограмма сравнения метрик")
    print("  - confusion_matrices_comparison.png : Матрицы ошибок рядом")
    print("  - roc_curves_comparison.png     : Сравнение ROC-кривых")
    print("  - confidence_comparison.png     : Сравнение распределений уверенности")
    print("  - mean_confidence_per_class.png : Средняя уверенность по классам")
    print("  - architecture_overview.png     : Обзор архитектуры и сравнение моделей")
    
    print("\n" + "=" * 70)
    print("ЗАВЕРШЕНО!")
    print("=" * 70)


if __name__ == "__main__":
    main()
