#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🚀 ЗАПУСК КЛАССИФИКАТОРА ТАКАГИ-СУГЕНО                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Способы запуска:                                                            ║
║    1. python run.py              - использует настройки из config.py         ║
║    2. python run.py --interactive - интерактивный режим с меню               ║
║    3. python run.py --mode boosting - быстрый выбор режима                   ║
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
    print("\n🔄 Загрузка классификатора...")
    
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
