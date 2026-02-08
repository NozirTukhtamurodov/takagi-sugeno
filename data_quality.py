"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             АНАЛИЗ И ОЧИСТКА КАЧЕСТВА ДАННЫХ ДЛЯ ТАКАГИ-СУГЕНО              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Проверяет train.csv на:                                                    ║
║    - NaN / Inf / -Inf                                                       ║
║    - Пустые строки                                                          ║
║    - Дубликаты                                                              ║
║    - Выбросы (IQR)                                                          ║
║    - Нулевую дисперсию (константные столбцы)                                ║
║    - Некорректные метки классов                                              ║
║  Автоматически исправляет найденные проблемы (среднее по классу).            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Использование:
    python data_quality.py                        # анализ train.csv
    python data_quality.py --file other.csv       # другой файл
    python data_quality.py --sep ";" --dec ","    # указать разделители
    python data_quality.py --fix                  # исправить и сохранить
    python data_quality.py --fix --output clean.csv  # сохранить в другой файл
"""

import argparse
import csv
import os
import sys
from collections import Counter
from copy import deepcopy
from typing import Optional


def parse_float(value: str, decimal: str = ".") -> Optional[float]:
    """Безопасно преобразовать строку в float."""
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in ("nan", "na", "n/a", "null", "none", "-", "?"):
        return None
    if decimal != ".":
        value = value.replace(decimal, ".")
    try:
        v = float(value)
        return v
    except (ValueError, OverflowError):
        return None


def is_finite(v: Optional[float]) -> bool:
    """Проверить, что значение конечное (не NaN, не Inf)."""
    if v is None:
        return False
    import math
    return math.isfinite(v)


def read_csv_raw(filepath: str, sep: str = ",", decimal: str = ".", has_header: bool = False):
    """
    Прочитать CSV и вернуть:
      header (list[str] или None), rows (list[list[str]]).
    """
    encoding = None
    try:
        with open(filepath, "rb") as f:
            bom = f.read(4)
            if bom[:2] == b"\xff\xfe":
                encoding = "utf-16-le"
            elif bom[:2] == b"\xfe\xff":
                encoding = "utf-16-be"
            elif bom[:3] == b"\xef\xbb\xbf":
                encoding = "utf-8-sig"
    except Exception:
        pass

    rows = []
    header = None
    with open(filepath, "r", encoding=encoding or "utf-8") as f:
        reader = csv.reader(f, delimiter=sep)
        for i, row in enumerate(reader):
            if i == 0 and has_header:
                header = row
                continue
            if row:
                rows.append(row)
    return header, rows


def analyze_and_fix(
    filepath: str,
    sep: str = ",",
    decimal: str = ".",
    has_header: bool = False,
    class_start_index: int = 1,
    fix: bool = False,
    output: Optional[str] = None,
    iqr_multiplier: float = 3.0,
):
    """Основная функция анализа и исправления данных."""
    import math

    print("=" * 70)
    print("АНАЛИЗ КАЧЕСТВА ДАННЫХ")
    print("=" * 70)
    print(f"  Файл:       {filepath}")
    print(f"  Разделитель: {repr(sep)}")
    print(f"  Десятичный:  {repr(decimal)}")
    print(f"  Заголовок:   {'Да' if has_header else 'Нет'}")
    print()

    if not os.path.exists(filepath):
        print(f"ОШИБКА: Файл '{filepath}' не найден!")
        sys.exit(1)

    header, rows = read_csv_raw(filepath, sep, decimal, has_header)
    n_rows = len(rows)

    if n_rows == 0:
        print("ОШИБКА: Файл пуст!")
        sys.exit(1)

    n_cols = max(len(r) for r in rows)
    n_features = n_cols - 1  # последний столбец — класс

    print(f"  Строк:       {n_rows}")
    print(f"  Столбцов:    {n_cols} ({n_features} признаков + 1 класс)")
    print()

    # ------------------------------------------------------------------
    # 1. Парсинг данных
    # ------------------------------------------------------------------
    data = []  # list[list[Optional[float]]]  — признаки
    labels = []  # list[Optional[int]] — метки
    parse_errors = []

    for row_idx, row in enumerate(rows):
        if len(row) != n_cols:
            parse_errors.append(
                (row_idx + 1, f"Неправильное число столбцов: {len(row)} вместо {n_cols}")
            )
            # Дополняем/обрезаем
            while len(row) < n_cols:
                row.append("")
            row = row[:n_cols]

        feature_vals = []
        for col_idx in range(n_features):
            v = parse_float(row[col_idx], decimal)
            feature_vals.append(v)
        data.append(feature_vals)

        # Класс
        label_str = row[-1].strip()
        try:
            label_val = int(float(label_str))
            labels.append(label_val - class_start_index)  # нормализуем к 0-based
        except (ValueError, OverflowError):
            labels.append(None)
            parse_errors.append((row_idx + 1, f"Некорректная метка класса: '{label_str}'"))

    # ------------------------------------------------------------------
    # 2. Сбор статистики
    # ------------------------------------------------------------------
    stats = {
        "nan_count": 0,
        "inf_count": 0,
        "parse_fail_count": 0,
        "empty_rows": 0,
        "duplicate_rows": 0,
        "bad_labels": 0,
        "constant_cols": [],
        "outlier_cells": 0,
        "nan_cells": [],   # (row, col)
        "inf_cells": [],   # (row, col)
        "outlier_cells_list": [],
    }

    # --- NaN / Inf / parse failures ---
    for r in range(n_rows):
        for c in range(n_features):
            v = data[r][c]
            if v is None:
                stats["nan_count"] += 1
                stats["nan_cells"].append((r, c))
            elif math.isinf(v):
                stats["inf_count"] += 1
                stats["inf_cells"].append((r, c))

    # --- Пустые строки (все признаки = None) ---
    for r in range(n_rows):
        if all(v is None for v in data[r]):
            stats["empty_rows"] += 1

    # --- Некорректные метки ---
    valid_labels = [l for l in labels if l is not None]
    stats["bad_labels"] = sum(1 for l in labels if l is None)
    unique_labels = sorted(set(valid_labels))

    # --- Дубликаты ---
    seen = {}
    for r in range(n_rows):
        key = tuple(data[r]) + (labels[r],)
        if key in seen:
            stats["duplicate_rows"] += 1
        else:
            seen[key] = r

    # --- Константные столбцы ---
    for c in range(n_features):
        col_vals = [data[r][c] for r in range(n_rows) if is_finite(data[r][c])]
        if len(col_vals) > 0 and len(set(col_vals)) == 1:
            stats["constant_cols"].append(c)

    # --- Выбросы (IQR) ---
    col_q1 = [0.0] * n_features
    col_q3 = [0.0] * n_features
    col_iqr = [0.0] * n_features
    col_lower = [0.0] * n_features
    col_upper = [0.0] * n_features

    for c in range(n_features):
        col_vals = sorted(v for r in range(n_rows) for v in [data[r][c]] if is_finite(v))
        if len(col_vals) < 4:
            col_lower[c] = float("-inf")
            col_upper[c] = float("inf")
            continue
        q1_idx = len(col_vals) // 4
        q3_idx = 3 * len(col_vals) // 4
        col_q1[c] = col_vals[q1_idx]
        col_q3[c] = col_vals[q3_idx]
        col_iqr[c] = col_q3[c] - col_q1[c]
        col_lower[c] = col_q1[c] - iqr_multiplier * col_iqr[c]
        col_upper[c] = col_q3[c] + iqr_multiplier * col_iqr[c]

    for r in range(n_rows):
        for c in range(n_features):
            v = data[r][c]
            if is_finite(v) and (v < col_lower[c] or v > col_upper[c]):
                stats["outlier_cells"] += 1
                stats["outlier_cells_list"].append((r, c, v))

    # ------------------------------------------------------------------
    # 3. Отчёт
    # ------------------------------------------------------------------
    print("-" * 70)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    print("-" * 70)

    total_issues = 0

    def report(label, count, detail=""):
        nonlocal total_issues
        total_issues += count
        status = "OK" if count == 0 else "ПРОБЛЕМА"
        icon = "[+]" if count == 0 else "[!]"
        msg = f"  {icon} {label}: {count}"
        if detail and count > 0:
            msg += f"  ({detail})"
        print(msg)

    report("NaN / пустые значения", stats["nan_count"])
    report("Inf / -Inf значения", stats["inf_count"])
    report("Полностью пустые строки", stats["empty_rows"])
    report("Некорректные метки класса", stats["bad_labels"])
    report("Дубликаты строк", stats["duplicate_rows"])
    report(
        "Константные столбцы (нулевая дисперсия)",
        len(stats["constant_cols"]),
        f"столбцы: {stats['constant_cols']}" if stats["constant_cols"] else "",
    )
    report(
        f"Выбросы (IQR x{iqr_multiplier})",
        stats["outlier_cells"],
    )

    if len(parse_errors) > 0:
        total_issues += len(parse_errors)
        print(f"  [!] Ошибки парсинга: {len(parse_errors)}")
        for row_num, msg in parse_errors[:10]:
            print(f"       Строка {row_num}: {msg}")
        if len(parse_errors) > 10:
            print(f"       ... и ещё {len(parse_errors) - 10}")

    print()

    # --- Распределение классов ---
    print("-" * 70)
    print("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    print("-" * 70)
    label_counts = Counter(valid_labels)
    for lbl in sorted(label_counts.keys()):
        original_lbl = lbl + class_start_index
        pct = label_counts[lbl] / len(valid_labels) * 100
        bar = "#" * int(pct / 2)
        print(f"  Класс {original_lbl}: {label_counts[lbl]:>5} образцов ({pct:5.1f}%) {bar}")
    if stats["bad_labels"] > 0:
        print(f"  ???:     {stats['bad_labels']:>5} образцов с некорректной меткой")
    print()

    # --- Базовая статистика по столбцам ---
    print("-" * 70)
    print("СТАТИСТИКА ПРИЗНАКОВ")
    print("-" * 70)
    print(f"  {'Призн':>5} | {'min':>10} | {'max':>10} | {'mean':>10} | {'std':>10} | {'NaN':>4} | {'Inf':>4} | {'Выбр':>4}")
    print("  " + "-" * 68)
    for c in range(n_features):
        col_vals = [data[r][c] for r in range(n_rows) if is_finite(data[r][c])]
        nan_c = sum(1 for r in range(n_rows) if data[r][c] is None)
        inf_c = sum(1 for r in range(n_rows) if data[r][c] is not None and math.isinf(data[r][c]))
        out_c = sum(1 for rr, cc, _ in stats["outlier_cells_list"] if cc == c)

        if col_vals:
            mn = min(col_vals)
            mx = max(col_vals)
            avg = sum(col_vals) / len(col_vals)
            var = sum((x - avg) ** 2 for x in col_vals) / max(len(col_vals) - 1, 1)
            std = var ** 0.5
        else:
            mn = mx = avg = std = float("nan")
        print(
            f"  {c:>5} | {mn:>10.3f} | {mx:>10.3f} | {avg:>10.3f} | {std:>10.3f} | {nan_c:>4} | {inf_c:>4} | {out_c:>4}"
        )
    print()

    # ------------------------------------------------------------------
    # 4. Итог
    # ------------------------------------------------------------------
    if total_issues == 0:
        print("=" * 70)
        print("ДАННЫЕ ЧИСТЫЕ — проблем не обнаружено!")
        print("=" * 70)
        if fix:
            print("Исправления не требуются.")
        return

    print("=" * 70)
    print(f"НАЙДЕНО ПРОБЛЕМ: {total_issues}")
    print("=" * 70)

    if not fix:
        print("\nДля автоматического исправления запустите с флагом --fix:")
        print(f"  python data_quality.py --fix")
        return

    # ------------------------------------------------------------------
    # 5. Исправление
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("АВТОМАТИЧЕСКОЕ ИСПРАВЛЕНИЕ")
    print("-" * 70)

    fixed_data = deepcopy(data)
    fixed_labels = list(labels)
    fix_log = []

    # 5a. Вычисляем среднее по классу для каждого признака
    class_means = {}
    for lbl in unique_labels:
        means = []
        for c in range(n_features):
            class_vals = [
                data[r][c]
                for r in range(n_rows)
                if labels[r] == lbl and is_finite(data[r][c])
            ]
            if class_vals:
                means.append(sum(class_vals) / len(class_vals))
            else:
                # Fallback — среднее по всему столбцу
                all_vals = [data[r][c] for r in range(n_rows) if is_finite(data[r][c])]
                means.append(sum(all_vals) / len(all_vals) if all_vals else 0.0)
        class_means[lbl] = means

    # Глобальное среднее для строк с неизвестным классом
    global_means = []
    for c in range(n_features):
        all_vals = [data[r][c] for r in range(n_rows) if is_finite(data[r][c])]
        global_means.append(sum(all_vals) / len(all_vals) if all_vals else 0.0)

    # Мода меток для строк с некорректной меткой
    if valid_labels:
        label_mode = Counter(valid_labels).most_common(1)[0][0]
    else:
        label_mode = 0

    # 5b. Исправление NaN / Inf -> среднее по классу
    fixed_nan = 0
    fixed_inf = 0
    for r in range(n_rows):
        lbl = fixed_labels[r]
        means = class_means.get(lbl, global_means)
        for c in range(n_features):
            v = fixed_data[r][c]
            if v is None:
                fixed_data[r][c] = means[c]
                fixed_nan += 1
                fix_log.append(f"  Строка {r+1}, столбец {c}: NaN -> {means[c]:.4f} (среднее по классу)")
            elif math.isinf(v):
                fixed_data[r][c] = means[c]
                fixed_inf += 1
                fix_log.append(f"  Строка {r+1}, столбец {c}: Inf -> {means[c]:.4f} (среднее по классу)")

    # 5c. Исправление некорректных меток -> мода
    fixed_labels_count = 0
    for r in range(n_rows):
        if fixed_labels[r] is None:
            fixed_labels[r] = label_mode
            fixed_labels_count += 1
            original_label = label_mode + class_start_index
            fix_log.append(f"  Строка {r+1}: метка -> {original_label} (мода)")

    # 5d. Удаление полностью пустых строк
    rows_to_remove = set()
    for r in range(n_rows):
        if all(data[r][c] is None for c in range(n_features)) and labels[r] is None:
            rows_to_remove.add(r)
    if rows_to_remove:
        fix_log.append(f"  Удалено полностью пустых строк: {len(rows_to_remove)}")

    # 5e. Удаление дубликатов
    seen_keys = set()
    for r in range(n_rows):
        if r in rows_to_remove:
            continue
        key = tuple(fixed_data[r]) + (fixed_labels[r],)
        if key in seen_keys:
            rows_to_remove.add(r)
        else:
            seen_keys.add(key)
    removed_dups = len(rows_to_remove) - stats["empty_rows"]

    # Печать лога (до 20 записей)
    for line in fix_log[:20]:
        print(line)
    if len(fix_log) > 20:
        print(f"  ... и ещё {len(fix_log) - 20} исправлений")

    print()
    print(f"  Исправлено NaN:            {fixed_nan}")
    print(f"  Исправлено Inf:            {fixed_inf}")
    print(f"  Исправлено меток:          {fixed_labels_count}")
    print(f"  Удалено пустых строк:      {stats['empty_rows']}")
    print(f"  Удалено дубликатов:        {removed_dups}")

    total_fixed = fixed_nan + fixed_inf + fixed_labels_count + len(rows_to_remove)

    if total_fixed == 0:
        print()
        print("=" * 70)
        print("Исправляемых проблем не обнаружено (выбросы не исправляются автоматически).")
        print("Файл не изменён.")
        print("=" * 70)
        return

    # ------------------------------------------------------------------
    # 6. Сохранение (только если были реальные изменения)
    # ------------------------------------------------------------------
    output_path = output or filepath
    backup_path = output_path + ".bak"

    # Собираем итоговые данные
    final_rows = []
    for r in range(n_rows):
        if r in rows_to_remove:
            continue
        row_vals = []
        for c in range(n_features):
            v = fixed_data[r][c]
            if v is not None:
                # Сохраняем формат: если целое — без точки, иначе с нужным числом знаков
                if v == int(v) and abs(v) < 1e9:
                    row_vals.append(str(int(v)))
                else:
                    row_vals.append(f"{v:.4f}".rstrip("0").rstrip("."))
            else:
                row_vals.append("")
        # Метка класса (возвращаем к исходной нумерации)
        lbl = fixed_labels[r]
        if lbl is not None:
            row_vals.append(str(lbl + class_start_index))
        else:
            row_vals.append("")
        final_rows.append(row_vals)

    # Бэкап оригинала (только если перезаписываем)
    if output_path == filepath and os.path.exists(filepath):
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"\n  Бэкап оригинала: {backup_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=sep)
        if header:
            writer.writerow(header)
        for row_vals in final_rows:
            # Восстанавливаем десятичный разделитель
            if decimal != ".":
                row_vals = [v.replace(".", decimal) for v in row_vals]
            writer.writerow(row_vals)

    print(f"\n  Сохранено:  {output_path} ({len(final_rows)} строк)")
    print()
    print("=" * 70)
    print("ИСПРАВЛЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Анализ и очистка качества данных для классификатора Такаги-Сугено"
    )
    parser.add_argument("--file", default=None, help="Путь к CSV файлу (по умолчанию из config.py)")
    parser.add_argument("--sep", default=None, help="Разделитель столбцов")
    parser.add_argument("--dec", default=None, help="Десятичный разделитель")
    parser.add_argument("--header", action="store_true", help="Файл содержит заголовок")
    parser.add_argument("--class-start", type=int, default=None, help="Начальный индекс классов (0 или 1)")
    parser.add_argument("--fix", action="store_true", help="Исправить найденные проблемы")
    parser.add_argument("--output", default=None, help="Путь для сохранения исправленного файла")
    parser.add_argument("--iqr", type=float, default=3.0, help="Множитель IQR для обнаружения выбросов (по умолчанию 3.0)")

    args = parser.parse_args()

    # Попробуем загрузить параметры из config.py
    filepath = args.file
    sep = args.sep
    dec = args.dec
    has_header = args.header
    class_start = args.class_start

    try:
        import config as cfg
        if filepath is None:
            filepath = getattr(cfg, "DATA_FILE", "train.csv")
        if sep is None:
            sep = getattr(cfg, "DATA_SEPARATOR", ",")
        if dec is None:
            dec = getattr(cfg, "DATA_DECIMAL", ".")
        if not has_header:
            has_header = getattr(cfg, "HAS_HEADER", False)
        if class_start is None:
            class_start = getattr(cfg, "CLASS_START_INDEX", 1)
    except ImportError:
        if filepath is None:
            filepath = "train.csv"
        if sep is None:
            sep = ","
        if dec is None:
            dec = "."
        if class_start is None:
            class_start = 1

    analyze_and_fix(
        filepath=filepath,
        sep=sep,
        decimal=dec,
        has_header=has_header,
        class_start_index=class_start,
        fix=args.fix,
        output=args.output,
        iqr_multiplier=args.iqr,
    )


if __name__ == "__main__":
    main()
