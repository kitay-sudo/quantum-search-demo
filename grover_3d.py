# grover_3d.py - 3D визуализация квантового куба
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import winsound
import threading

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

def grover(N, target, iterations=None, record=False, slow_mode=False):
    amps = init_state(N)
    if iterations is None:
        iterations = int(round((math.pi/4) * math.sqrt(N)))

    # Медленный режим: больше промежуточных кадров
    if slow_mode:
        iterations = iterations * 3  # Увеличиваем в 3 раза

    probs_over_time = []
    amps_over_time = []

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

def play_quantum_sound(frequency, duration=100):
    def play():
        try:
            winsound.Beep(int(frequency), duration)
        except:
            pass
    threading.Thread(target=play, daemon=True).start()

def prob_to_frequency(prob, min_freq=200, max_freq=2000):
    return min_freq + (max_freq - min_freq) * prob

def create_3d_quantum_cube(N, target, probs_over_time, amps_over_time, save_gif=False):
    """
    3D визуализация квантового куба с вращением
    """
    # Для 3D куба используем 8 состояний (2^3 = 8 углов куба)
    # Если N != 8, берём первые 8 или дополняем
    if N < 8:
        print("ПРЕДУПРЕЖДЕНИЕ: N < 8, дополняем нулями")
        probs_padded = [np.pad(p, (0, 8-N), 'constant') for p in probs_over_time]
    elif N > 8:
        probs_padded = [p[:8] for p in probs_over_time]
    else:
        probs_padded = probs_over_time

    # Координаты 8 углов куба
    cube_coords = np.array([
        [0, 0, 0],  # 0: (000)
        [1, 0, 0],  # 1: (001)
        [0, 1, 0],  # 2: (010)
        [1, 1, 0],  # 3: (011)
        [0, 0, 1],  # 4: (100)
        [1, 0, 1],  # 5: (101)
        [0, 1, 1],  # 6: (110)
        [1, 1, 1],  # 7: (111)
    ])

    # Рёбра куба (для отрисовки линий)
    edges = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7]
    ]

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#0a0a1a')

    # 3D куб (БОЛЬШОЙ, слева)
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_3d.set_facecolor('#16213e')
    ax_3d.set_xlabel('X (Кубит 0)', fontsize=13, color='cyan', fontweight='bold')
    ax_3d.set_ylabel('Y (Кубит 1)', fontsize=13, color='cyan', fontweight='bold')
    ax_3d.set_zlabel('Z (Кубит 2)', fontsize=13, color='cyan', fontweight='bold')
    ax_3d.set_title('3D Квантовый Куб', fontsize=18, fontweight='bold', color='lime', pad=20)
    ax_3d.set_xlim(-0.5, 1.5)
    ax_3d.set_ylim(-0.5, 1.5)
    ax_3d.set_zlim(-0.5, 1.5)
    ax_3d.tick_params(colors='white')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.grid(True, alpha=0.3, color='cyan')

    # Правая панель: только текст
    ax_info = fig.add_subplot(1, 2, 2)
    ax_info.axis('off')
    ax_info.set_facecolor('#0a0a1a')

    # Рисуем рёбра куба (ярче и толще)
    edge_lines = []
    for edge in edges:
        p1, p2 = cube_coords[edge[0]], cube_coords[edge[1]]
        line, = ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                          'cyan', alpha=0.5, linewidth=2)
        edge_lines.append(line)

    # Сферы в углах куба (КРУПНЫЕ!)
    scatter = ax_3d.scatter(cube_coords[:, 0], cube_coords[:, 1], cube_coords[:, 2],
                           s=500, c='cyan', alpha=0.9, edgecolors='white', linewidth=3,
                           depthshade=True)

    # Подписи состояний (КРУПНЕЕ!)
    text_labels = []
    for i, coord in enumerate(cube_coords):
        label = f'{i}\n({i:03b})'
        color = 'yellow'  # Изначально все жёлтые, зелёными станут в процессе
        txt = ax_3d.text(coord[0], coord[1], coord[2] + 0.15, label,
                  fontsize=14, ha='center', color=color, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color))
        text_labels.append(txt)

    # Информационная панель БЕЗ рамок - просто текст
    info_text = ax_info.text(0.1, 0.95, '', transform=ax_info.transAxes,
                            fontsize=16, verticalalignment='top', family='monospace',
                            color='cyan', fontweight='normal')

    # Счётчик угла вращения
    rotation_angle = [0]

    def update(frame):
        # Получаем вероятности для текущего кадра
        probs = probs_padded[frame]

        # Обновляем размер и цвет сфер (КРУПНЕЕ!)
        sizes = [max(800, p * 5000) for p in probs]  # размер от вероятности
        scatter._sizes = sizes

        # Яркие цвета: синий -> желтый -> красный
        temp_colors = []
        for i, p in enumerate(probs):
            if i == target:
                # Целевая вершина: от синего к зелёному по мере роста вероятности
                if p < 0.3:
                    # Сначала синяя как все
                    r = 0
                    g = p * 3  # постепенно зеленеет
                    b = 1 - p * 2
                elif p < 0.7:
                    # Становится зелёной
                    r = 0
                    g = 0.5 + p * 0.5
                    b = max(0, 1 - p * 2)
                else:
                    # Яркая зелёная при высокой вероятности
                    r = 0
                    g = 0.8 + p * 0.2
                    b = 0
                temp_colors.append(np.array([r, g, b, 0.95]))
            else:
                # Остальные: от синего к жёлтому к красному
                if p < 0.5:
                    # Синий -> Желтый
                    r = p * 2
                    g = p * 2
                    b = 1 - p * 2
                else:
                    # Желтый -> Красный
                    r = 1
                    g = 1 - (p - 0.5) * 2
                    b = 0
                temp_colors.append(np.array([r, g, b, 0.9]))
        scatter._facecolors = np.array(temp_colors)

        # Обновляем цвет меток
        for i, txt in enumerate(text_labels):
            if i == target and probs[i] > 0.6:
                # Метка цели становится зелёной при высокой вероятности
                txt.set_color('lime')
                txt.get_bbox_patch().set_edgecolor('lime')
            else:
                txt.set_color('yellow')
                txt.get_bbox_patch().set_edgecolor('yellow')

        # Медленное вращение камеры
        rotation_angle[0] = (rotation_angle[0] + 1.5) % 360
        ax_3d.view_init(elev=25, azim=rotation_angle[0])

        # Обновляем информацию с красивыми отступами
        max_idx = np.argmax(probs)
        max_prob = probs[max_idx]

        status = "НАЙДЕНО!" if max_idx == target and max_prob > 0.5 else "ПОИСК..."
        status_color = "lime" if max_idx == target and max_prob > 0.5 else "yellow"

        info = f"""╔════════════════════════════════════╗
║     КВАНТОВЫЙ АЛГОРИТМ ГРОВЕРА     ║
╚════════════════════════════════════╝

  Итерация:      {frame:3d} / {len(probs_over_time)-1}

  Цель:          {target} (двоичное: {target:03b})
                 X={target & 1}, Y={(target>>1)&1}, Z={(target>>2)&1}

  Макс. вер-ть:  {max_idx} ({max_prob:.4f})
  Статус:        {status}

  Камера:        {rotation_angle[0]:.0f}°

════════════════════════════════════════

КАК ЧИТАТЬ:

  РАЗМЕР СФЕРЫ = Вероятность
  ─────────────────────────────────
  • Маленькая → Низкая вероятность
  • Большая   → Высокая вероятность

  ЦВЕТ СФЕРЫ = Значение вероятности
  ─────────────────────────────────
  • Синяя     → Начало (низкая)
  • Жёлтая    → Средняя
  • Красная   → Высокая
  • ЗЕЛЁНАЯ   → ЦЕЛЬ (target={target})

════════════════════════════════════════
"""
        info_text.set_text(info)

        # Звук
        target_prob = probs[target] if target < len(probs) else 0
        freq = prob_to_frequency(target_prob)
        play_quantum_sound(freq, duration=150)

        # Победный звук
        if frame == len(probs_over_time) - 1 and max_idx == target:
            threading.Timer(0.3, lambda: play_quantum_sound(1500, 100)).start()
            threading.Timer(0.5, lambda: play_quantum_sound(2000, 150)).start()

        return [scatter] + edge_lines + [info_text]

    # Создаём анимацию (МЕДЛЕННЕЕ для лучшего просмотра!)
    anim = FuncAnimation(fig, update, frames=len(probs_over_time),
                        interval=1000, blit=False, repeat=True)  # 1000ms = 1 сек на кадр

    # Сохранение в GIF
    if save_gif:
        print("\nСОХРАНЕНИЕ АНИМАЦИИ В GIF...")
        print("Это может занять 30-60 секунд...")
        writer = PillowWriter(fps=2)
        filename = 'quantum_cube_3d.gif'
        anim.save(filename, writer=writer)
        print(f"GIF сохранён: {filename}")
        print(f"Размер файла: {get_file_size(filename)}")

    # plt.tight_layout()  # Отключено из-за ручного позиционирования
    return fig, anim

def get_file_size(filename):
    import os
    size = os.path.getsize(filename)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

if __name__ == "__main__":
    print("=" * 60)
    print(">>> 3D КВАНТОВЫЙ КУБ - АЛГОРИТМ ГРОВЕРА <<<")
    print("=" * 60)

    # Для 3D куба используем 3 кубита (8 состояний)
    n = 3
    N = 2**n
    target = random.randrange(N)

    print(f"\nПараметры:")
    print(f"   Количество кубитов: {n}")
    print(f"   Размер пространства: {N} (углов куба)")
    print(f"   Цель (десят.): {target}")
    print(f"   Цель (двоич.):  {target:03b}")
    print(f"   Цель (x,y,z):    ({target&1}, {(target>>1)&1}, {(target>>2)&1})")

    optimal_iterations = int(round((math.pi/4) * math.sqrt(N)))
    print(f"   Оптимальных итераций: {optimal_iterations}")

    print(f"\nЗапуск алгоритма Гровера...")
    print(f"   (Медленный режим для лучшего просмотра)")
    # МЕДЛЕННЫЙ РЕЖИМ: в 3 раза больше кадров!
    amps, probs_over_time, amps_over_time = grover(N, target, iterations=None, record=True, slow_mode=True)
    probs = np.abs(amps)**2

    found = int(np.argmax(probs))
    print(f"\nРезультаты:")
    print(f"   Алгоритм нашёл: {found}")
    print(f"   Правильный ответ: {target}")
    print(f"   Статус: {'УСПЕХ' if found == target else 'ОШИБКА'}")
    print(f"   Вероятность: {probs[found]:.4f}")

    # Спрашиваем про сохранение GIF
    print("\n" + "=" * 60)
    try:
        save_choice = input("Сохранить анимацию в GIF? (y/n): ").strip().lower()
        save_gif = save_choice in ['y', 'yes', 'д', 'да', '1']
    except EOFError:
        save_gif = False
        print("n (автоматически)")

    print(f"\nГенерация 3D визуализации...")
    print(f"   Яркие цвета [OK]")
    print(f"   Крупные сферы [OK]")
    print(f"   Медленное вращение [OK]")
    fig, anim = create_3d_quantum_cube(N, target, probs_over_time, amps_over_time, save_gif=save_gif)

    play_quantum_sound(800, 100)

    print(f"\n>>> 3D КУБ ВРАЩАЕТСЯ! <<<")
    print(f"   СО ЗВУКОМ (только при просмотре, не в GIF)!")
    print(f"   Закройте окно для завершения.")
    print(f"   Кадров: {len(probs_over_time)} (медленный режим)")
    print("=" * 60)

    plt.show()
