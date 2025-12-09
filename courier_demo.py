# courier_demo.py - Курьер ищет дом: Квантовый vs Классический поиск
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import winsound
import threading

# === АЛГОРИТМ ГРОВЕРА ===
def init_state(N):
    return np.ones(N, dtype=float) / math.sqrt(N)

def oracle(amps, target):
    out = amps.copy()
    out[target] *= -1.0
    return out

def diffusion(amps):
    mean = np.mean(amps)
    return 2*mean - amps

def grover(N, target, record=True):
    amps = init_state(N)
    iterations = int(round((math.pi/4) * math.sqrt(N)))
    history = [np.abs(amps)**2] if record else []

    for i in range(iterations):
        amps = oracle(amps, target)
        amps = diffusion(amps)
        if record:
            history.append(np.abs(amps)**2)

    return amps, history, iterations

def play_sound(freq, duration=100):
    def play():
        try:
            winsound.Beep(int(freq), duration)
        except:
            pass
    threading.Thread(target=play, daemon=True).start()

# === АНИМИРОВАННАЯ ВИЗУАЛИЗАЦИЯ ===
def courier_visualization_animated(N=100, target=None):
    if target is None:
        target = random.randint(0, N-1)

    # Запускаем алгоритм Гровера
    amps, quantum_history, quantum_iters = grover(N, target)

    # Классический поиск
    classical_checks = list(range(N))
    random.shuffle(classical_checks)
    found_at_classical = classical_checks.index(target) + 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f'КУРЬЕР ИЩЕТ ДОМ №{target} ИЗ {N} ДОМОВ',
                 fontsize=18, fontweight='bold', color='lime')

    # === КАРТА УЛИЦЫ ===
    ax_street = axes[0]
    ax_street.set_title('КАРТА УЛИЦЫ (10x10 домов)', fontsize=14, color='cyan', fontweight='bold')
    ax_street.set_xlim(-1, 11)
    ax_street.set_ylim(-1, 11)
    ax_street.set_facecolor('#0a0a1a')
    ax_street.set_xticks([])
    ax_street.set_yticks([])

    # Рисуем сетку домов
    grid_size = 10
    houses = []
    house_texts = []
    for i in range(N):
        row = i // grid_size
        col = i % grid_size
        rect = plt.Rectangle((col, row), 0.8, 0.8,
                            facecolor='gray', edgecolor='white', linewidth=1)
        ax_street.add_patch(rect)
        houses.append(rect)

        # Номер дома
        txt = ax_street.text(col+0.4, row+0.4, str(i),
                      ha='center', va='center', fontsize=8, color='white')
        house_texts.append(txt)

    # Курьер (кружок)
    courier_quantum = plt.Circle((0.4, 0.4), 0.3, color='lime', alpha=0, zorder=10)
    courier_classical = plt.Circle((0.4, 0.4), 0.3, color='orange', alpha=0, zorder=9)
    ax_street.add_patch(courier_quantum)
    ax_street.add_patch(courier_classical)

    # === СТАТИСТИКА (ПРОСТОЙ ТЕКСТ) ===
    ax_stats = axes[1]
    ax_stats.axis('off')
    ax_stats.set_facecolor('#0a0a1a')

    info_text = ax_stats.text(0.05, 0.95, '', transform=ax_stats.transAxes,
                              fontsize=13, verticalalignment='top',
                              family='monospace', color='cyan',
                              fontweight='normal')

    # Переменные анимации
    # Создаем временную шкалу: каждая итерация Гровера = несколько проверок классического
    checks_per_iteration = max(1, found_at_classical // len(quantum_history))
    max_frames = len(quantum_history) * 3  # Делаем анимацию более плавной

    def update(frame):
        # КВАНТОВЫЙ - обновляем по итерациям (медленнее)
        q_iter = min(frame // 3, len(quantum_history) - 1)

        # КЛАССИЧЕСКИЙ - обновляем по проверкам (быстрее)
        c_checked = min(frame * checks_per_iteration, found_at_classical)

        # Сбрасываем все дома в серый цвет
        for i, house in enumerate(houses):
            house.set_facecolor('gray')
            house.set_alpha(1.0)
            house.set_edgecolor('white')
            house.set_linewidth(1)

        # КВАНТОВЫЙ КУРЬЕР - подсвечивает все дома сразу (суперпозиция)
        current_probs = quantum_history[q_iter]
        max_prob_idx = np.argmax(current_probs)

        # Показываем квантовые вероятности на домах
        for i in range(N):
            prob = current_probs[i]
            if prob > 0.05:  # Показываем только значимые вероятности
                # Интенсивность зеленого зависит от вероятности
                houses[i].set_facecolor('lime')
                houses[i].set_alpha(min(0.9, prob * 2))
                if i == target and q_iter >= len(quantum_history) - 1:
                    houses[i].set_edgecolor('yellow')
                    houses[i].set_linewidth(3)

        # Позиция квантового курьера (на самом вероятном доме)
        q_row = max_prob_idx // grid_size
        q_col = max_prob_idx % grid_size
        courier_quantum.set_center((q_col + 0.4, q_row + 0.4))
        courier_quantum.set_alpha(0.9 if q_iter < len(quantum_history) - 1 else 0)

        # КЛАССИЧЕСКИЙ КУРЬЕР - последовательная проверка
        if c_checked < found_at_classical:
            for check_idx in range(c_checked):
                house_idx = classical_checks[check_idx]
                # Подсвечиваем уже проверенные дома оранжевым
                houses[house_idx].set_facecolor('orange')
                houses[house_idx].set_alpha(0.3)

            # Текущий проверяемый дом
            if c_checked > 0:
                current_check = classical_checks[c_checked - 1]
                c_row = current_check // grid_size
                c_col = current_check % grid_size
                courier_classical.set_center((c_col + 0.4, c_row + 0.4))
                courier_classical.set_alpha(0.8)
        else:
            # Нашли дом классически
            houses[target].set_facecolor('orange')
            houses[target].set_alpha(0.9)
            houses[target].set_edgecolor('yellow')
            houses[target].set_linewidth(3)
            courier_classical.set_alpha(0)

        # Целевой дом - золотая звездочка
        target_row = target // grid_size
        target_col = target % grid_size

        # СТАТИСТИКА
        quantum_done = q_iter >= len(quantum_history) - 1
        classical_done = c_checked >= found_at_classical

        speedup = found_at_classical / max(q_iter + 1, 1)
        theoretical_speedup = math.sqrt(N)

        stats = f"""
РЕЗУЛЬТАТЫ ПОИСКА
{'═'*55}

🎯 ЦЕЛЬ: Дом №{target} из {N} домов

{'═'*55}
🟢 КВАНТОВЫЙ КУРЬЕР (Зеленый)
{'═'*55}
Итераций:         {q_iter + 1} / {len(quantum_history)}
Метод:            Алгоритм Гровера
Принцип:          Суперпозиция (все дома сразу)
Сложность:        O(√N) ≈ {int(math.sqrt(N))} проверок
Статус:           {'✓ НАЙДЕН!' if quantum_done else '⏳ ПОИСК...'}
Вероятность:      {current_probs[target]:.1%}

{'═'*55}
🟠 КЛАССИЧЕСКИЙ КУРЬЕР (Оранжевый)
{'═'*55}
Проверено:        {c_checked} / {found_at_classical}
Метод:            Последовательный обход
Принцип:          По одному дому
Сложность:        O(N) ≈ {N//2} проверок (среднее)
Статус:           {'✓ НАЙДЕН!' if classical_done else '⏳ ПОИСК...'}

{'═'*55}
📊 СРАВНЕНИЕ
{'═'*55}
Ускорение:        {speedup:.1f}x
Теоретическое:    {theoretical_speedup:.1f}x (√{N})
Экономия:         {found_at_classical - (q_iter+1)} проверок

{'─'*55}
💡 Квантовый проверяет ВСЕ дома одновременно!
   Классический - только ПО ОДНОМУ.
{'═'*55}
"""
        info_text.set_text(stats)

        # Звук
        if q_iter < len(quantum_history):
            freq = 300 + current_probs[target] * 1500
            play_sound(freq, 60)

        return [courier_quantum, courier_classical, info_text] + houses

    # Анимация с паузой между итерациями
    anim = FuncAnimation(fig, update, frames=max_frames,
                        interval=800, blit=False, repeat=True)  # 800ms = пауза между итерациями

    plt.tight_layout()

    # Звук старта
    play_sound(600, 100)

    return fig, anim, quantum_iters, found_at_classical

if __name__ == "__main__":
    N = 100
    target = random.randint(0, N-1)

    print("="*60)
    print("КУРЬЕР ИЩЕТ ДОМ - Квантовый vs Классический")
    print("="*60)
    print(f"Цель: дом №{target} из {N}")
    print(f"Алгоритм Гровера: ~{int(round((math.pi/4) * math.sqrt(N)))} итераций")
    print(f"Ожидаемое ускорение: {math.sqrt(N):.0f}x")
    print("\nЗагрузка анимации...")

    fig, anim, q_iters, c_checks = courier_visualization_animated(N, target)

    speedup = c_checks / q_iters
    print(f"Готово! Ускорение: {speedup:.1f}x ({q_iters} vs {c_checks} проверок)")

    # Предложение сохранить GIF
    print("\n" + "="*60)
    save_choice = input("Сохранить анимацию в GIF? (y/n): ").strip().lower()

    if save_choice in ['да', 'yes', 'y', 'д']:
        print("\n🎬 Сохранение GIF...")
        print("⚠️  Это может занять 30-60 секунд, пожалуйста подождите...")

        try:
            from matplotlib.animation import PillowWriter

            # Создаем writer для GIF (со звуком нельзя, только визуал)
            writer = PillowWriter(fps=1.2)  # ~800ms на кадр

            filename = f'courier_demo_target{target}.gif'
            anim.save(filename, writer=writer, dpi=80)

            import os
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"\n✓ GIF сохранен: {filename}")
            print(f"  Размер: {file_size:.1f} MB")
            print(f"  ⚠️  ЗВУК в GIF не сохраняется (только при просмотре в Python)")

        except Exception as e:
            print(f"\n✗ Ошибка при сохранении GIF: {e}")
            print("  Попробуйте установить: pip install pillow")

    print("\n" + "="*60)
    print("Показываю анимацию (со звуком!)...")
    print("Закройте окно для завершения.")
    print("="*60)

    plt.show()
