# menu.py - Меню запуска программ
import subprocess
import sys
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("=" * 60)
        print("         КВАНТОВЫЙ МОЗГ - МЕНЮ ПРОГРАММ")
        print("=" * 60)
        print()
        print("Выберите программу:")
        print()
        print("  1. Основная 2D визуализация (16 состояний)")
        print("  2. 3D вращающийся куб (8 состояний)")
        print("  3. Сравнение квантовый vs классический")
        print("  4. Курьер ищет дом (100 домов)")
        print("  5. Запустить ВСЁ одновременно")
        print("  6. Выйти")
        print()

        choice = input("Ваш выбор (1-6): ").strip()

        if choice == "1":
            print("\nЗапуск 2D визуализации...")
            subprocess.run([sys.executable, "grover_demo.py"])
        elif choice == "2":
            print("\nЗапуск 3D куба...")
            subprocess.run([sys.executable, "grover_3d.py"])
        elif choice == "3":
            print("\nЗапуск сравнения...")
            subprocess.run([sys.executable, "quantum_comparison.py"])
        elif choice == "4":
            print('\nЗапуск "Курьер ищет дом"...')
            subprocess.run([sys.executable, "courier_demo.py"])
        elif choice == "5":
            print("\nЗапуск ВСЕХ программ одновременно...")
            subprocess.Popen([sys.executable, "grover_demo.py"])
            subprocess.Popen([sys.executable, "grover_3d.py"])
            subprocess.Popen([sys.executable, "quantum_comparison.py"])
            subprocess.Popen([sys.executable, "courier_demo.py"])
            print("\nВсе программы запущены!")
            input("\nНажмите Enter для возврата в меню...")
        elif choice == "6":
            print("\nГотово!")
            break
        else:
            print("\nНеверный выбор! Попробуйте снова...")
            input("Нажмите Enter...")

if __name__ == "__main__":
    main()
