#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🚀 ЗАПУСК КЛАССИФИКАТОРА ТАКАГИ-СУГЕНО                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Способы запуска:                                                            ║
║    1. python run.py              - использует настройки из config.py         ║
║    2. python run.py --interactive - интерактивный режим с меню               ║
║    3. python run.py --mode boosting - быстрый выбор режима                   ║
║    4. python run.py --check      - проверка качества данных                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import argparse

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """Печать красивого баннера."""
    print("\n" + "═" * 70)
    print("║" + " " * 68 + "║")
    print("║" + "  🔮 НЕЧЁТКИЙ КЛАССИФИКАТОР ТАКАГИ-СУГЕНО".center(66) + "  ║")
    print("║" + "  с нейтрософским расширением".center(66) + "  ║")
    print("║" + " " * 68 + "║")
    print("═" * 70)


def print_menu():
    """Печать интерактивного меню."""
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    ВЫБЕРИТЕ РЕЖИМ РАБОТЫ                        │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  [1] 🎯 Одиночная модель     - базовый режим                    │")
    print("│  [2] 👥 Ансамбль             - несколько моделей (soft voting)  │")
    print("│  [3] 🚀 Бустинг              - адаптивное обучение (РЕКОМЕНД.)  │")
    print("│  [4] 🏛️  Иерархическая       - двухуровневая классификация      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print("│  [5] ⚙️  Настроить параметры  - редактировать config.py         │")
    print("│  [6] 📖 Справка              - описание режимов                 │")
    print("│  [0] ❌ Выход                                                   │")
    print("└─────────────────────────────────────────────────────────────────┘")


def print_help():
    """Печать справки о режимах."""
    print("\n" + "─" * 70)
    print("📖 ОПИСАНИЕ РЕЖИМОВ:")
    print("─" * 70)
    print("""
🎯 ОДИНОЧНАЯ МОДЕЛЬ (Single)
   Базовый классификатор Такаги-Сугено. Быстрое обучение.
   Лучше для: небольших задач, быстрого прототипирования.

👥 АНСАМБЛЬ (Ensemble)
   Несколько моделей с разными параметрами, усреднение предсказаний.
   Лучше для: повышения стабильности, уменьшения переобучения.

🚀 БУСТИНГ (Boosting) — РЕКОМЕНДУЕТСЯ
   Последовательное обучение с акцентом на трудных примерах.
   Лучше для: максимальной точности, сложных задач.

🏛️ ИЕРАРХИЧЕСКАЯ (Hierarchical)
   Сначала классификация в группы, затем внутри групп.
   Лучше для: задач с большим количеством похожих классов.
""")
    print("─" * 70)


def check_data(verbose: bool = True) -> tuple:
    """
    Проверка качества данных.
    
    Returns:
        tuple: (is_ok, nan_percentage, details_dict)
            - is_ok: True если данные можно использовать
            - nan_percentage: процент повреждённых ячеек
            - details: словарь с деталями проблем
    """
    import pandas as pd
    import numpy as np
    import config
    
    data_file = config.DATA_FILE
    separator = getattr(config, 'DATA_SEPARATOR', ';')
    decimal = getattr(config, 'DATA_DECIMAL', ',')
    has_header = getattr(config, 'HAS_HEADER', False)
    
    header = 0 if has_header else None
    
    # Определяем engine
    file_ext = os.path.splitext(data_file)[1].lower()
    engine = 'python' if file_ext in ['.txt', '.data', '.dat'] else None
    
    # Автоопределение кодировки
    encoding = None
    try:
        with open(data_file, 'rb') as f:
            first_bytes = f.read(4)
            if first_bytes[:2] == b'\xff\xfe':
                encoding = 'utf-16-le'
            elif first_bytes[:2] == b'\xfe\xff':
                encoding = 'utf-16-be'
            elif first_bytes[:3] == b'\xef\xbb\xbf':
                encoding = 'utf-8-sig'
    except:
        pass
    
    if verbose:
        print("\n" + "═" * 70)
        print("🔍 ПРОВЕРКА КАЧЕСТВА ДАННЫХ")
        if encoding:
            print(f"   📝 Кодировка: {encoding}")
        print("═" * 70)
        print(f"   📁 Файл: {data_file}")
    
    try:
        data = pd.read_csv(data_file, sep=separator, decimal=decimal, header=header, engine=engine, encoding=encoding)
    except Exception as e:
        if verbose:
            print(f"\n   ❌ ОШИБКА: Не удалось прочитать файл!")
            print(f"      {e}")
        return False, 100.0, {"error": str(e)}
    
    if verbose:
        print(f"   📊 Размер: {data.shape[0]} строк × {data.shape[1]} столбцов")
    
    # Проверяем тип последнего столбца (классы)
    last_col = data.iloc[:, -1]
    is_text_classes = not pd.api.types.is_numeric_dtype(last_col)
    
    if is_text_classes and verbose:
        print(f"   📝 Классы: текстовые ({last_col.nunique()} уникальных)")
    elif verbose:
        print(f"   🔢 Классы: числовые ({last_col.nunique()} уникальных)")
    
    # Конвертируем в числовой формат (ТОЛЬКО признаки, без последнего столбца)
    data_features = data.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    
    # Анализ NaN (только для признаков)
    nan_per_column = data_features.isna().sum()
    total_nan = nan_per_column.sum()
    total_cells = data_features.shape[0] * data_features.shape[1]
    nan_percentage = total_nan / total_cells * 100 if total_cells > 0 else 0
    rows_with_nan = data_features.isna().any(axis=1).sum()
    
    details = {
        "total_cells": total_cells,
        "total_nan": total_nan,
        "nan_percentage": nan_percentage,
        "rows_with_nan": rows_with_nan,
        "problem_columns": {},
        "is_text_classes": is_text_classes
    }
    
    if verbose:
        print(f"\n   📈 РЕЗУЛЬТАТЫ ПРОВЕРКИ:")
        print(f"   ────────────────────────────────────────")
    
    if total_nan == 0:
        if verbose:
            print(f"   ✅ ОТЛИЧНО! Все данные корректны.")
            print(f"   ✅ Можно начинать обучение.")
        return True, 0.0, details
    
    # Есть проблемы
    if verbose:
        print(f"   ⚠️  Всего ячеек:        {total_cells}")
        print(f"   ❌ Некорректных:        {total_nan} ({nan_percentage:.2f}%)")
        print(f"   📋 Строк с проблемами:  {rows_with_nan}")
    
    # Детали по столбцам
    problem_cols = nan_per_column[nan_per_column > 0]
    if len(problem_cols) > 0 and verbose:
        print(f"\n   📊 ПРОБЛЕМНЫЕ СТОЛБЦЫ:")
        print(f"   ────────────────────────────────────────")
        
    for col in problem_cols.index:
        col_nan = problem_cols[col]
        col_pct = col_nan / len(data) * 100
        details["problem_columns"][col] = {"nan": col_nan, "percent": col_pct}
        
        if verbose:
            if col_pct > 50:
                severity = "🔴 КРИТИЧНО"
            elif col_pct > 10:
                severity = "🟡 ВНИМАНИЕ"
            else:
                severity = "🟢 Незначит."
            print(f"      Столбец {col:2d}: {col_nan:5d} NaN ({col_pct:5.1f}%) {severity}")
    
    # Вердикт
    if verbose:
        print(f"\n   📋 ВЕРДИКТ:")
        print(f"   ────────────────────────────────────────")
    
    if nan_percentage > 20:
        if verbose:
            print(f"   🚨 КРИТИЧНО: {nan_percentage:.1f}% данных повреждены!")
            print(f"   ❌ ОБУЧЕНИЕ ЗАБЛОКИРОВАНО")
            print(f"\n   💡 РЕКОМЕНДАЦИИ:")
            print(f"      1. Проверьте исходный файл данных")
            print(f"      2. Возможные причины:")
            print(f"         - Неправильная кодировка файла")
            print(f"         - Смешанные десятичные разделители")
            print(f"         - Повреждение при экспорте из Excel")
            print(f"      3. Исправьте файл и запустите проверку снова:")
            print(f"         python run.py --check")
        return False, nan_percentage, details
    
    elif nan_percentage > 5:
        if verbose:
            print(f"   ⚠️  ВНИМАНИЕ: {nan_percentage:.1f}% данных повреждены.")
            print(f"   ⚠️  Проблемные строки будут удалены при обучении.")
            print(f"   ✅ Обучение возможно, но рекомендуется исправить данные.")
        return True, nan_percentage, details
    
    else:
        if verbose:
            print(f"   ℹ️  Незначительные проблемы: {nan_percentage:.2f}%")
            print(f"   ✅ Обучение возможно (проблемные строки будут удалены)")
        return True, nan_percentage, details


def show_current_config():
    """Показать текущую конфигурацию."""
    try:
        import config
        print("\n" + "─" * 70)
        print("⚙️  ТЕКУЩАЯ КОНФИГУРАЦИЯ (из config.py):")
        print("─" * 70)
        print(f"   📁 Файл данных:     {config.DATA_FILE}")
        print(f"   📊 PCA:             {'Вкл (' + str(config.PCA_VARIANCE*100) + '%)' if config.USE_PCA else 'Выкл'}")
        print(f"   📈 Тестовая выборка: {config.TEST_SIZE*100:.0f}%")
        print()
        
        # Показываем информацию о данных
        try:
            import pandas as pd
            import numpy as np
            import os
            data_file = config.DATA_FILE
            separator = getattr(config, 'DATA_SEPARATOR', ';')
            decimal = getattr(config, 'DATA_DECIMAL', ',')
            has_header = getattr(config, 'HAS_HEADER', False)
            class_start = getattr(config, 'CLASS_START_INDEX', 1)
            
            header = 0 if has_header else None
            
            # Определяем engine в зависимости от расширения файла
            file_ext = os.path.splitext(data_file)[1].lower()
            if file_ext in ['.txt', '.data', '.dat']:
                engine = 'python'
            else:
                engine = None  # auto
            
            # Автоопределение кодировки
            encoding = None
            try:
                with open(data_file, 'rb') as f:
                    first_bytes = f.read(4)
                    if first_bytes[:2] == b'\xff\xfe':
                        encoding = 'utf-16-le'
                    elif first_bytes[:2] == b'\xfe\xff':
                        encoding = 'utf-16-be'
                    elif first_bytes[:3] == b'\xef\xbb\xbf':
                        encoding = 'utf-8-sig'
            except:
                pass
            
            data = pd.read_csv(data_file, sep=separator, decimal=decimal, header=header, nrows=100, engine=engine, encoding=encoding)
            n_features = data.shape[1] - 1
            
            # Читаем только столбец классов для подсчёта
            data_full = pd.read_csv(data_file, sep=separator, decimal=decimal, header=header, usecols=[data.shape[1]-1], engine=engine, encoding=encoding)
            
            # Проверяем тип классов (текст или число)
            last_col = data_full.iloc[:, 0]
            if pd.api.types.is_numeric_dtype(last_col):
                y = last_col.values.astype(int) - class_start
                n_classes = int(np.max(y)) + 1
            else:
                n_classes = last_col.nunique()
            n_samples = len(data_full)
            
            print(f"   📊 Данные из файла: {data_file}")
            print(f"   📈 Примеров:        {n_samples}")
            print(f"   🔢 Признаков:       {n_features}")
            print(f"   🏷️  Классов:         {n_classes}")
            print()
        except Exception as e:
            print(f"   ⚠️  Не удалось прочитать данные: {e}")
            print()
        
        print(f"   🔧 Функций принадл.: {config.N_MFS}")
        print(f"   📋 Макс. правил:    {config.MAX_RULES}")
        print(f"   🔒 Регуляризация:   {config.REGULARIZATION}")
        print(f"   🌡️  Температура:     {config.TEMPERATURE}")
        print()
        
        mode = "Не выбран"
        if config.MODE_BOOSTING:
            mode = f"🚀 Бустинг ({config.BOOSTING_ROUNDS} раундов)"
        elif config.MODE_ENSEMBLE:
            mode = f"👥 Ансамбль ({config.ENSEMBLE_N_ESTIMATORS} моделей)"
        elif config.MODE_HIERARCHICAL:
            mode = f"🏛️ Иерархическая ({config.HIERARCHICAL_N_GROUPS} групп)"
        elif config.MODE_SINGLE:
            mode = "🎯 Одиночная модель"
        print(f"   🎮 Режим:           {mode}")
        print("─" * 70)
    except ImportError:
        print("⚠️  Файл config.py не найден!")


def update_config_mode(mode: str):
    """Обновить режим в config.py."""
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Сбрасываем все режимы
    content = content.replace("MODE_SINGLE = True", "MODE_SINGLE = False")
    content = content.replace("MODE_ENSEMBLE = True", "MODE_ENSEMBLE = False")
    content = content.replace("MODE_BOOSTING = True", "MODE_BOOSTING = False")
    content = content.replace("MODE_HIERARCHICAL = True", "MODE_HIERARCHICAL = False")
    
    # Включаем выбранный режим
    mode_map = {
        "single": "MODE_SINGLE",
        "ensemble": "MODE_ENSEMBLE",
        "boosting": "MODE_BOOSTING",
        "hierarchical": "MODE_HIERARCHICAL"
    }
    
    if mode in mode_map:
        var_name = mode_map[mode]
        content = content.replace(f"{var_name} = False", f"{var_name} = True")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Режим изменён на: {mode}")
        return True
    return False


def run_classifier():
    """Запуск классификатора с текущей конфигурацией."""
    
    # ===== ПРОВЕРКА ДАННЫХ ПЕРЕД ОБУЧЕНИЕМ =====
    print("\n🔄 Проверка данных перед обучением...")
    is_ok, nan_pct, details = check_data(verbose=True)
    
    if not is_ok:
        print("\n" + "═" * 70)
        print("❌ ОБУЧЕНИЕ ОТМЕНЕНО: данные не прошли проверку!")
        print("═" * 70)
        print("\n💡 Исправьте данные и запустите снова.")
        return False
    
    print("\n" + "═" * 70)
    print("✅ Данные проверены, начинаю обучение...")
    print("═" * 70)
    
    # Перезагружаем конфиг
    import importlib
    import config
    importlib.reload(config)
    
    # Импортируем и настраиваем главный модуль
    import takagi_sugeno_optimized as ts
    
    # Обновляем HYPER_CONFIG из config.py
    ts.HYPER_CONFIG = ts.HyperConfig(
        n_mfs_min=config.N_MFS,
        n_mfs_max=config.N_MFS + 2,
        n_mfs_divisor=20,
        max_rules_min=config.MAX_RULES - 50,
        max_rules_max=config.MAX_RULES + 100,
        max_rules_multiplier=2,
        regularization=config.REGULARIZATION,
        temperature=config.TEMPERATURE,
        overlap_factor=config.OVERLAP_FACTOR,
        multiclass_threshold=50,
        use_pca=config.USE_PCA,
        pca_variance=config.PCA_VARIANCE,
        pca_n_components=config.PCA_N_COMPONENTS,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        use_ensemble=config.MODE_ENSEMBLE,
        n_estimators=config.ENSEMBLE_N_ESTIMATORS,
        ensemble_diversity=config.ENSEMBLE_DIVERSITY,
        use_hierarchical=config.MODE_HIERARCHICAL,
        n_groups=config.HIERARCHICAL_N_GROUPS,
        use_boosting=config.MODE_BOOSTING,
        n_boosting_rounds=config.BOOSTING_ROUNDS,
        boosting_learning_rate=config.BOOSTING_LEARNING_RATE
    )
    
    # Запускаем main
    ts.main()


def interactive_mode():
    """Интерактивный режим с меню."""
    print_banner()
    
    while True:
        show_current_config()
        print_menu()
        
        try:
            choice = input("\n👉 Ваш выбор: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 До свидания!")
            sys.exit(0)
        
        if choice == "1":
            update_config_mode("single")
            run_classifier()
            break
        elif choice == "2":
            update_config_mode("ensemble")
            run_classifier()
            break
        elif choice == "3":
            update_config_mode("boosting")
            run_classifier()
            break
        elif choice == "4":
            update_config_mode("hierarchical")
            run_classifier()
            break
        elif choice == "5":
            config_path = os.path.join(os.path.dirname(__file__), "config.py")
            print(f"\n📝 Откройте файл для редактирования:\n   {config_path}")
            print("\n   Или используйте команду:")
            print(f"   code {config_path}")
            input("\n   Нажмите Enter после редактирования...")
        elif choice == "6":
            print_help()
            input("\nНажмите Enter для продолжения...")
        elif choice == "0":
            print("\n👋 До свидания!")
            sys.exit(0)
        else:
            print("\n⚠️  Неверный выбор. Попробуйте снова.")


def main():
    """Главная точка входа."""
    parser = argparse.ArgumentParser(
        description="🔮 Нечёткий классификатор Такаги-Сугено",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run.py                    # Запуск с настройками из config.py
  python run.py -i                 # Интерактивный режим с меню
  python run.py --mode boosting    # Быстрый запуск бустинга
  python run.py --mode ensemble    # Быстрый запуск ансамбля
  python run.py --config           # Показать текущую конфигурацию
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Запустить интерактивный режим с меню"
    )
    
    parser.add_argument(
        "-m", "--mode",
        choices=["single", "ensemble", "boosting", "hierarchical"],
        help="Выбрать режим работы"
    )
    
    parser.add_argument(
        "-c", "--config",
        action="store_true",
        help="Показать текущую конфигурацию"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Проверить качество данных (без обучения)"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не генерировать графики"
    )
    
    args = parser.parse_args()
    
    # Показать конфигурацию
    if args.config:
        print_banner()
        show_current_config()
        sys.exit(0)
    
    # Проверка данных
    if args.check:
        print_banner()
        is_ok, nan_pct, details = check_data(verbose=True)
        print("\n" + "═" * 70)
        if is_ok:
            print("✅ Можно запускать обучение: python run.py")
        else:
            print("❌ Исправьте данные перед обучением!")
        print("═" * 70)
        sys.exit(0 if is_ok else 1)
    
    # Интерактивный режим
    if args.interactive:
        interactive_mode()
        return
    
    # Быстрый выбор режима
    if args.mode:
        print_banner()
        update_config_mode(args.mode)
    
    # Отключение графиков
    if args.no_plots:
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace("GENERATE_PLOTS = True", "GENERATE_PLOTS = False")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("📊 Генерация графиков отключена")
    
    # Запуск классификатора
    print_banner()
    show_current_config()
    
    confirm = input("\n🚀 Запустить с этими настройками? [Y/n]: ").strip().lower()
    if confirm in ["", "y", "yes", "да", "д"]:
        run_classifier()
    else:
        print("\n💡 Используйте 'python run.py -i' для интерактивного режима")
        print("   или отредактируйте config.py")


if __name__ == "__main__":
    main()
