import pygame
import sys

# Инициализация Pygame
pygame.init()

# Размеры окна
WIDTH, HEIGHT = 1268, 840

# Цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Рисование линий")

# Список для хранения координат линий
lines = []

# Основной цикл программы
running = True
drawing = False
line_array = []
screen.fill(WHITE)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # Начало рисования линии при нажатии левой кнопки мыши
                x1, y1 = event.pos
                drawing = True
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                # Обновление экрана при движении мыши для отображения линии
                screen.fill(WHITE)
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line
                    pygame.draw.line(screen, RED, (x1_l, y1_l), (x2_l, y2_l), 2)
                x2, y2 = event.pos
                pygame.draw.line(screen, RED, (x1, y1), (x2, y2), 2)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and drawing:
                # Завершение рисования линии при отпускании левой кнопки мыши
                x2, y2 = event.pos
                lines.append((x1, y1, x2, y2))

                
                pygame.display.flip()
                drawing = False

    # Обновление экрана
    pygame.display.flip()

# Вывод координат линий
for i, line in enumerate(lines):
    print(f"[{line[0]}, {line[1]}, {line[2]}, {line[3]}],", end=" ")

# Завершение работы Pygame
pygame.quit()
sys.exit()
