# quantum_comparison.py - Сравнение квантового и классического поиска
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
import winsound
import threading
import time

def init_state(N):
    amp = np.ones(N, dtype=float) / math.sqrt(N)
    return amp

def oracle(amps, target):
    out = amps.copy()
    out[target] *= -1.0
    return out

def diffusion(amps):
    mean = np.mean(amps)
    return 2*mean - amps

def grover(N, target, iterations=None, record=False):
    amps = init_state(N)
    if iterations is None:
        iterations = int(round((math.pi/4) * math.sqrt(N)))
    probs_over_time = []

    if record:
        probs_over_time.append(np.abs(amps)**2)

    for i in range(iterations):
        amps = oracle(amps, target)
        amps = diffusion(amps)
        if record:
            probs_over_time.append(np.abs(amps)**2)

    return amps, probs_over_time

def classical_search_sequential(N, target, record=False):
    """
    Классический последовательный поиск
    """
    checked = []
    for i in range(N):
        checked.append(i)
        if i == target:
            break
    return checked

def play_sound(frequency, duration=50):
    def play():
        try:
            winsound.Beep(int(frequency), duration)
        except:
            pass
    threading.Thread(target=play, daemon=True).start()

def compare_quantum_vs_classical(N, target, probs_over_time, save_gif=False):
    """
    Визуальное сравнение квантового и классического поиска
    """
    grid_size = int(math.ceil(math.sqrt(N)))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('КВАНТОВЫЙ vs КЛАССИЧЕСКИЙ ПОИСК', fontsize=16, fontweight='bold')

    # Квантовый поиск (слева)
    ax_quantum = fig.add_subplot(2, 3, 1)
    ax_quantum.set_title('КВАНТОВЫЙ (Гровер)', fontsize=12, color='lime')
    ax_quantum.set_xticks([])
    ax_quantum.set_yticks([])

    # Классический поиск (справа)
    ax_classical = fig.add_subplot(2, 3, 2)
    ax_classical.set_title('КЛАССИЧЕСКИЙ (Последовательный)', fontsize=12, color='orange')
    ax_classical.set_xticks([])
    ax_classical.set_yticks([])

    # Сравнение скорости
    ax_compare = fig.add_subplot(2, 3, 3)
    ax_compare.set_title('СРАВНЕНИЕ СКОРОСТИ', fontsize=12)
    ax_compare.set_xlabel('Размер задачи (N)')
    ax_compare.set_ylabel('Операций')
    ax_compare.set_yscale('log')
    ax_compare.grid(True, alpha=0.3)

    # График вероятности (квантовый)
    ax_prob_q = fig.add_subplot(2, 3, 4)
    ax_prob_q.set_title('Вероятность (Квантовый)', fontsize=11)
    ax_prob_q.set_xlabel('Итерация')
    ax_prob_q.set_ylabel('P(цель)')
    ax_prob_q.set_xlim(0, len(probs_over_time)-1)
    ax_prob_q.set_ylim(0, 1)
    ax_prob_q.grid(True, alpha=0.3)

    # Столбчатая диаграмма
    ax_bars = fig.add_subplot(2, 3, 5)
    ax_bars.set_title('Распределение вероятностей', fontsize=11)
    ax_bars.set_xlabel('Состояние')
    ax_bars.set_ylabel('Вероятность')
    ax_bars.set_ylim(0, 1)
    ax_bars.grid(True, alpha=0.3)

    # Статистика
    ax_stats = fig.add_subplot(2, 3, 6)
    ax_stats.axis('off')

    # График сравнения сложности
    sizes = [2**i for i in range(1, 13)]  # от 2 до 4096
    classical_ops = sizes  # O(N)
    quantum_ops = [math.sqrt(n) * (math.pi/4) for n in sizes]  # O(√N)
    ax_compare.plot(sizes, classical_ops, 'o-', color='orange', linewidth=2, label='Классический O(N)', markersize=6)
    ax_compare.plot(sizes, quantum_ops, 's-', color='lime', linewidth=2, label='Квантовый O(√N)', markersize=6)
    ax_compare.legend()
    ax_compare.axvline(x=N, color='red', linestyle='--', alpha=0.5, label=f'N={N}')

    # Пиксельная сетка для квантового
    pixel_grid_q = np.zeros((grid_size, grid_size))
    im_q = ax_quantum.imshow(pixel_grid_q, cmap='hot', interpolation='nearest', vmin=0, vmax=1)

    # Пиксельная сетка для классического
    pixel_grid_c = np.zeros((grid_size, grid_size))
    im_c = ax_classical.imshow(pixel_grid_c, cmap='gray', interpolation='nearest', vmin=0, vmax=1)

    # Рамка вокруг целевого элемента
    target_row = target // grid_size
    target_col = target % grid_size
    rect_q = patches.Rectangle((target_col-0.5, target_row-0.5), 1, 1,
                               linewidth=3, edgecolor='lime', facecolor='none')
    rect_c = patches.Rectangle((target_col-0.5, target_row-0.5), 1, 1,
                               linewidth=3, edgecolor='orange', facecolor='none')
    ax_quantum.add_patch(rect_q)
    ax_classical.add_patch(rect_c)

    # График вероятности
    target_probs = [p[target] for p in probs_over_time]
    prob_line, = ax_prob_q.plot([], [], 'lime', linewidth=3, marker='o', markersize=5)
    ax_prob_q.fill_between(range(len(target_probs)), target_probs, alpha=0.3, color='lime')

    # Столбцы
    bars = ax_bars.bar(range(min(N, 32)), np.zeros(min(N, 32)), color='cyan', alpha=0.7)
    if N <= 32:
        ax_bars.axvline(x=target, color='lime', linestyle='--', linewidth=2)

    # Текст статистики
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                              fontsize=10, verticalalignment='top', family='monospace')

    # Классический поиск - последовательность проверок
    classical_checks = list(range(N))
    random.shuffle(classical_checks)  # случайный порядок проверки
    target_index_classical = classical_checks.index(target)

    # Переменные для анимации
    total_frames = max(len(probs_over_time), target_index_classical + 2)

    def update(frame):
        # КВАНТОВЫЙ ПОИСК
        q_frame = min(frame, len(probs_over_time) - 1)
        probs = probs_over_time[q_frame]

        # Обновляем пиксели квантового
        pixel_grid_q = np.zeros((grid_size, grid_size))
        for i, prob in enumerate(probs):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                pixel_grid_q[row, col] = prob
        im_q.set_array(pixel_grid_q)

        # КЛАССИЧЕСКИЙ ПОИСК
        pixel_grid_c = np.zeros((grid_size, grid_size))
        classical_found = False

        if frame <= target_index_classical:
            # Помечаем уже проверенные элементы
            for i in range(frame + 1):
                if i <= target_index_classical:
                    idx = classical_checks[i]
                    row = idx // grid_size
                    col = idx % grid_size
                    if row < grid_size and col < grid_size:
                        if idx == target:
                            pixel_grid_c[row, col] = 1.0  # нашли!
                            classical_found = True
                        else:
                            pixel_grid_c[row, col] = 0.3  # проверили, не то
        else:
            # Классический уже нашёл
            idx = target
            row = idx // grid_size
            col = idx % grid_size
            pixel_grid_c[row, col] = 1.0
            classical_found = True

        im_c.set_array(pixel_grid_c)

        # Обновляем график вероятности
        prob_line.set_data(range(q_frame + 1), target_probs[:q_frame + 1])

        # Обновляем столбцы
        if N <= 32:
            for i, (bar, prob) in enumerate(zip(bars, probs[:len(bars)])):
                bar.set_height(prob)
                if i == target:
                    bar.set_color('lime')
                else:
                    bar.set_color('cyan')

        # Статистика
        quantum_iterations = q_frame
        classical_iterations = min(frame + 1, target_index_classical + 1)
        quantum_done = q_frame >= len(probs_over_time) - 1 and np.argmax(probs) == target

        speedup = classical_iterations / max(quantum_iterations, 1)

        stats = f"""
СТАТИСТИКА
{'='*35}

КВАНТОВЫЙ:
  Итераций: {quantum_iterations}
  Статус:   {'НАЙДЕН!' if quantum_done else 'ПОИСК...'}
  Вер-ть:   {probs[target]:.3f}

КЛАССИЧЕСКИЙ:
  Проверок:  {classical_iterations}
  Статус:    {'НАЙДЕН!' if classical_found else 'ПОИСК...'}

{'='*35}
УСКОРЕНИЕ: {speedup:.1f}x

Теоретическое: {math.sqrt(N):.1f}x
{'='*35}

N = {N}
Квантовый: ~{int(round((math.pi/4) * math.sqrt(N)))} итер
Классич:   ~{N//2} проверок (сред)
        """
        stats_text.set_text(stats)

        # Звук
        if q_frame < len(target_probs):
            freq = 200 + target_probs[q_frame] * 1800
            play_sound(freq, 50)

        return [im_q, im_c, prob_line] + list(bars) + [stats_text]

    # Анимация
    anim = FuncAnimation(fig, update, frames=total_frames,
                        interval=500, blit=False, repeat=True)

    # Сохранение в GIF
    if save_gif:
        print("\nСОХРАНЕНИЕ В GIF...")
        writer = PillowWriter(fps=2)
        filename = 'quantum_vs_classical.gif'
        anim.save(filename, writer=writer)
        print(f"GIF сохранён: {filename}")

    plt.tight_layout()
    return fig, anim

if __name__ == "__main__":
    print("=" * 60)
    print("СРАВНЕНИЕ: КВАНТОВЫЙ vs КЛАССИЧЕСКИЙ ПОИСК")
    print("=" * 60)

    # Выбор размера задачи
    print("\nВыберите размер задачи:")
    print("  1. Маленький   (16 элементов, 4 кубита)")
    print("  2. Средний     (64 элемента, 6 кубитов)")
    print("  3. Большой     (256 элементов, 8 кубитов)")
    print("  4. Огромный    (1024 элемента, 10 кубитов)")

    try:
        choice = input("\nВаш выбор (1-4): ").strip()
    except EOFError:
        choice = '1'
        print("1 (автоматически)")

    size_map = {
        '1': 4,
        '2': 6,
        '3': 8,
        '4': 10
    }

    n = size_map.get(choice, 4)
    N = 2**n
    target = random.randrange(N)

    print(f"\nПараметры:")
    print(f"   Кубитов: {n}")
    print(f"   Размер:  {N}")
    print(f"   Цель:    {target}")

    optimal = int(round((math.pi/4) * math.sqrt(N)))
    print(f"\n   Квантовый:     ~{optimal} итераций")
    print(f"   Классический:  ~{N//2} проверок (среднее)")
    print(f"   УСКОРЕНИЕ:     ~{math.sqrt(N):.1f}x")

    print(f"\nЗапуск алгоритма Гровера...")
    amps, probs_over_time = grover(N, target, iterations=None, record=True)
    probs = np.abs(amps)**2

    found = int(np.argmax(probs))
    print(f"   Найдено: {found}")
    print(f"   Правильно: {'ДА' if found == target else 'НЕТ'}")

    try:
        save_choice = input("\nСохранить GIF? (y/n): ").strip().lower()
        save_gif = save_choice in ['y', 'yes', 'д', 'да', '1']
    except EOFError:
        save_gif = False
        print("n (автоматически)")

    print(f"\nГенерация визуализации...")
    fig, anim = compare_quantum_vs_classical(N, target, probs_over_time, save_gif=save_gif)

    play_sound(800, 100)
    print(f"\nСмотрите анимацию СО ЗВУКОМ!")
    print(f"(ЗВУК не сохраняется в GIF, только при просмотре)")
    print("=" * 60)

    plt.show()
