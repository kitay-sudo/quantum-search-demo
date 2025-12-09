# grover_demo.py
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import winsound
import threading

def init_state(N):
    # равная амплитуда для всех состояний (возможно комплексная, но пока вещественная)
    amp = np.ones(N, dtype=float) / math.sqrt(N)
    return amp

def oracle(amps, target):
    # меняем знак амплитуды у правильного состояния
    out = amps.copy()
    out[target] *= -1.0
    return out

def diffusion(amps):
    # отражение относительно среднего: a -> 2*mean - a
    mean = np.mean(amps)
    return 2*mean - amps

def grover(N, target, iterations=None, record=False):
    amps = init_state(N)
    if iterations is None:
        iterations = int(round((math.pi/4) * math.sqrt(N)))
    probs_over_time = []
    amps_over_time = []

    # Сохраняем начальное состояние
    if record:
        probs_over_time.append(np.abs(amps)**2)
        amps_over_time.append(amps.copy())

    for i in range(iterations):
        amps = oracle(amps, target)
        amps = diffusion(amps)
        if record:
            probs_over_time.append(np.abs(amps)**2)
            amps_over_time.append(amps.copy())

    return amps, probs_over_time, amps_over_time

def top_k_probs(probs, k=5):
    idx = np.argsort(probs)[-k:][::-1]
    return list(zip(idx, probs[idx]))

def play_quantum_sound(frequency, duration=100):
    """
    Воспроизводит звук с заданной частотой
    frequency: частота в Гц (37-32767)
    duration: длительность в миллисекундах
    """
    def play():
        try:
            winsound.Beep(int(frequency), duration)
        except:
            pass
    threading.Thread(target=play, daemon=True).start()

def prob_to_frequency(prob, min_freq=200, max_freq=2000):
    """
    Конвертирует вероятность в частоту звука
    Чем выше вероятность, тем выше тон
    """
    return min_freq + (max_freq - min_freq) * prob

def visualize_quantum_brain(N, target, probs_over_time, amps_over_time):
    """
    Визуализация квантового мозга в виде пикселей
    """
    # Создаём сетку для пикселей
    grid_size = int(math.sqrt(N))
    if grid_size * grid_size < N:
        grid_size += 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('🧠 Квантовый Мозг - Алгоритм Гровера 🧠', fontsize=16, fontweight='bold')

    # Настройка осей
    ax_pixels = axes[0, 0]  # Пиксельная визуализация
    ax_bars = axes[0, 1]     # Столбчатая диаграмма
    ax_target = axes[1, 0]   # График целевой вероятности
    ax_amps = axes[1, 1]     # Амплитуды

    # Инициализация пиксельной сетки
    pixel_grid = np.zeros((grid_size, grid_size))
    im = ax_pixels.imshow(pixel_grid, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    ax_pixels.set_title('Квантовые состояния (пиксели)', fontsize=12)
    ax_pixels.set_xticks([])
    ax_pixels.set_yticks([])

    # Отмечаем целевое состояние
    target_row = target // grid_size
    target_col = target % grid_size
    rect = patches.Rectangle((target_col-0.5, target_row-0.5), 1, 1,
                             linewidth=3, edgecolor='lime', facecolor='none')
    ax_pixels.add_patch(rect)

    # Цветовая шкала
    cbar = plt.colorbar(im, ax=ax_pixels, fraction=0.046, pad=0.04)
    cbar.set_label('Вероятность', rotation=270, labelpad=15)

    # Столбчатая диаграмма
    bars = ax_bars.bar(range(N), probs_over_time[0], color='cyan', alpha=0.7)
    ax_bars.set_xlim(-1, min(N, 64))  # Показываем максимум 64 состояния
    ax_bars.set_ylim(0, 1)
    ax_bars.set_xlabel('Индекс состояния')
    ax_bars.set_ylabel('Вероятность')
    ax_bars.set_title('Распределение вероятностей', fontsize=12)
    ax_bars.axvline(x=target, color='lime', linestyle='--', linewidth=2, label=f'Target={target}')
    ax_bars.legend()
    ax_bars.grid(True, alpha=0.3)

    # График целевой вероятности
    iterations = len(probs_over_time)
    target_probs = [probs[target] for probs in probs_over_time]
    line_target, = ax_target.plot(range(iterations), target_probs,
                                  color='lime', linewidth=2, marker='o', markersize=4)
    ax_target.set_xlim(0, iterations-1)
    ax_target.set_ylim(0, 1)
    ax_target.set_xlabel('Итерация')
    ax_target.set_ylabel('Вероятность')
    ax_target.set_title(f'Рост вероятности target={target}', fontsize=12)
    ax_target.grid(True, alpha=0.3)
    ax_target.fill_between(range(iterations), target_probs, alpha=0.3, color='lime')

    # График амплитуд
    bars_amps = ax_amps.bar(range(N), amps_over_time[0], color='purple', alpha=0.7)
    ax_amps.set_xlim(-1, min(N, 64))
    ax_amps.set_ylim(-1, 1)
    ax_amps.set_xlabel('Индекс состояния')
    ax_amps.set_ylabel('Амплитуда')
    ax_amps.set_title('Амплитуды состояний', fontsize=12)
    ax_amps.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
    ax_amps.axvline(x=target, color='lime', linestyle='--', linewidth=2)
    ax_amps.grid(True, alpha=0.3)

    # Текстовая информация
    text_info = ax_pixels.text(0.02, 0.98, '', transform=ax_pixels.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                               color='lime', family='monospace')

    # Анимация
    def update(frame):
        # Обновляем пиксельную сетку
        pixel_grid = np.zeros((grid_size, grid_size))
        for i, prob in enumerate(probs_over_time[frame]):
            row = i // grid_size
            col = i % grid_size
            if row < grid_size and col < grid_size:
                pixel_grid[row, col] = prob
        im.set_array(pixel_grid)

        # Обновляем столбчатую диаграмму вероятностей
        for i, (bar, prob) in enumerate(zip(bars, probs_over_time[frame])):
            if i < len(bars):
                bar.set_height(prob)
                # Подсвечиваем целевое состояние
                if i == target:
                    bar.set_color('lime')
                else:
                    bar.set_color('cyan')

        # Обновляем столбчатую диаграмму амплитуд
        for i, (bar, amp) in enumerate(zip(bars_amps, amps_over_time[frame])):
            if i < len(bars_amps):
                bar.set_height(amp)
                if amp < 0:
                    bar.set_color('red')
                elif i == target:
                    bar.set_color('lime')
                else:
                    bar.set_color('purple')

        # Обновляем информацию
        max_idx = np.argmax(probs_over_time[frame])
        max_prob = probs_over_time[frame][max_idx]
        target_prob = probs_over_time[frame][target]

        # Статус поиска
        if frame == 0:
            status = '⏳ ИНИЦИАЛИЗАЦИЯ...'
        elif frame < len(probs_over_time) - 1:
            status = f'🔍 ПОИСК... (итерация {frame}/{len(probs_over_time)-1})'
        elif max_idx == target:
            status = '✓ НАЙДЕНО!'
        else:
            status = '✗ НЕ НАЙДЕНО'

        info_text = f'Итерация: {frame}/{len(probs_over_time)-1}\n'
        info_text += f'Цель: {target}\n'
        info_text += f'Макс. вер-ть: {max_idx} ({max_prob:.3f})\n'
        info_text += f'Вер-ть цели: {target_prob:.3f}\n'
        info_text += f'\nСтатус: {status}'
        text_info.set_text(info_text)

        # ЗВУКОВАЯ ИНДИКАЦИЯ!
        # Воспроизводим звук с частотой, зависящей от вероятности целевого состояния
        freq = prob_to_frequency(target_prob)
        play_quantum_sound(freq, duration=150)

        # Дополнительный "успешный" звук на последней итерации
        if frame == len(probs_over_time) - 1 and max_idx == target:
            threading.Timer(0.3, lambda: play_quantum_sound(1500, 100)).start()
            threading.Timer(0.5, lambda: play_quantum_sound(2000, 150)).start()

        return [im] + list(bars) + list(bars_amps) + [text_info]

    # Создаём анимацию (замедляем для демонстрации)
    anim = FuncAnimation(fig, update, frames=len(probs_over_time),
                        interval=800, blit=False, repeat=True)

    plt.tight_layout()
    return fig, anim

if __name__ == "__main__":
    print("=" * 60)
    print("KVANTOVYJ MOZG - ALGORITM GROVERA")
    print("=" * 60)

    n = 4                  # число кубитов -> N = 2^n (можешь менять)
    N = 2**n
    target = random.randrange(N)

    print(f"\nПараметры:")
    print(f"   Количество кубитов: {n}")
    print(f"   Размер пространства поиска: {N}")
    print(f"   Искомое значение (цель): {target}")

    optimal_iterations = int(round((math.pi/4) * math.sqrt(N)))
    print(f"   Оптимальное число итераций: {optimal_iterations}")

    print(f"\nЗапуск алгоритма Гровера...")
    amps, probs_over_time, amps_over_time = grover(N, target, iterations=None, record=True)
    probs = np.abs(amps)**2

    found = int(np.argmax(probs))
    print(f"\nРезультаты:")
    print(f"   Алгоритм нашёл: {found}")
    print(f"   Правильный ответ: {target}")
    print(f"   Статус: {'УСПЕХ' if found == target else 'ОШИБКА'}")
    print(f"   Вероятность найденного: {probs[found]:.4f}")

    print(f"\nТоп-5 вероятностей:")
    for idx, p in top_k_probs(probs, 5):
        marker = ">>" if idx == target else "  "
        print(f"   {marker} Состояние {idx:>3}: {p:.4f} {'#' * int(p*40)}")

    print(f"\nГенерация визуализации квантового мозга...")
    print(f"ЗВУК: Частота звука растёт вместе с вероятностью!")
    print(f"      Низкий тон -> низкая вероятность")
    print(f"      Высокий тон -> высокая вероятность")
    fig, anim = visualize_quantum_brain(N, target, probs_over_time, amps_over_time)

    print(f"\nВизуализация готова!")

    # Предложение сохранить GIF
    print("\n" + "="*60)
    save_choice = input("Сохранить анимацию в GIF? (y/n): ").strip().lower()

    if save_choice in ['da', 'yes', 'y', 'd', 'да', 'д']:
        print("\n🎬 Сохранение GIF...")
        print("⚠️  Это может занять 20-40 секунд, пожалуйста подождите...")

        try:
            from matplotlib.animation import PillowWriter

            writer = PillowWriter(fps=1.25)  # ~800ms на кадр

            filename = f'grover_brain_N{N}_target{target}.gif'
            anim.save(filename, writer=writer, dpi=80)

            import os
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            print(f"\n✓ GIF сохранён: {filename}")
            print(f"  Размер: {file_size:.1f} MB")
            print(f"  ⚠️  ЗВУК в GIF не сохраняется (только при просмотре в Python)")

        except Exception as e:
            print(f"\n✗ Ошибка при сохранении GIF: {e}")
            print("  Попробуйте установить: pip install pillow")

    print("\n" + "="*60)
    print("Показываю анимацию (со звуком!)...")
    print(f"Закройте окно графика для завершения программы.")
    print("=" * 60)

    # Начальный звук запуска
    play_quantum_sound(800, 100)

    plt.show()
