import numpy as np
import pygame
import math
import numpy as np
from pygame.math import Vector2
import math

def distance(x1, y1, x2, y2):
    # Используем теорему Пифагора для вычисления расстояния
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx**2 + dy**2)
    return distance

def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    line1_start = Vector2(x1, y1)
    line1_end = Vector2(x2, y2)
    line2_start = Vector2(x3, y3)
    line2_end = Vector2(x4, y4)

    line1_direction = line1_end - line1_start
    line2_direction = line2_end - line2_start

    if line1_direction.cross(line2_direction) != 0:
        intersection_point = line1_start + line1_direction * ((line2_start - line1_start).cross(line2_direction) / line1_direction.cross(line2_direction))
        
        # Проверка, находится ли точка пересечения внутри отрезков
        if (min(x1, x2) <= intersection_point.x <= max(x1, x2) and
            min(y1, y2) <= intersection_point.y <= max(y1, y2) and
            min(x3, x4) <= intersection_point.x <= max(x3, x4) and
            min(y3, y4) <= intersection_point.y <= max(y3, y4)):
            return intersection_point
            
    return None


def find_intersection_points(barrier_array, car_x, car_y, end_x, end_y):
    intersection_points = []
    for barrier in barrier_array:
        intersection_point = intersection(barrier[0], barrier[1], barrier[2], barrier[3], car_x, car_y, end_x, end_y)
        if intersection_point:
            intersection_points.append(intersection_point)
    distances_array = []
    for points in intersection_points:
        x1, y1 = points
        distances_array.append(distance(x1, y1, car_x, car_y))
    if distances_array:
        return min(distances_array)
    else:
        return None
     


# Создаем среду с двумя состояниями и двумя действиями
num_states = 3
num_actions = 3



# Инициализация Pygame
pygame.init()

# Размеры окна
WINDOW_WIDTH, WINDOW_HEIGHT = 1268, 840

# Создаем окно
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Движение по трассе")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)



# Создаем изображение объекта (прямоугольника)
car_width = 20
car_height = 20
car_color = BLACK
car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
pygame.draw.circle(car_surface, car_color, (20,20), car_width)

# Начальная позиция объекта
car_x = 369
car_y = 537

# Параметры движения и вращения
speed = 3  # Скорость движения объекта
rotation_speed = 5  # Скорость вращения объекта
car_angle = 180  # Угол направления движения в градусах

barrier_array = np.array([[20, 378, 64, 130], [64, 130, 445, 54], [445, 54, 856, 116], [856, 116, 1157, 307], [1157, 307, 1040, 649], [1040, 649, 689, 714], [689, 714, 460, 680], 
                          [460, 680, 187, 576], [20, 378, 187, 576],
                          [159, 344, 189, 200],[189, 200, 443, 150],[443, 150, 820, 234],[820, 234, 969, 345],[969, 345, 959, 500],[959, 500, 875, 556],[875, 556, 582, 560],
                          [582, 560, 159, 344]])  #

ray_length = 5000

# Основной цикл игры
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE) 
    # Управление движением и вращением (аналогично предыдущему коду)
    


    # Преобразование угла в радианы
    radians = math.radians(car_angle)

    # Рассчитываем новые координаты объекта в сторону угла
    car_x += int(speed * math.cos(radians))
    car_y -= int(speed * math.sin(radians))  # Уменьшаем, чтобы двигаться вверх, так как ось Y внизу в Pygame


    
    end_x1 = car_x + ray_length * math.cos(math.radians(car_angle))
    end_y1 = car_y - ray_length * math.sin(math.radians(car_angle))

    end_x2 = car_x + ray_length * math.cos(math.radians(car_angle-90))
    end_y2 = car_y - ray_length * math.sin(math.radians(car_angle-90))

    end_x3 = car_x + ray_length * math.cos(math.radians(car_angle+90))
    end_y3 = car_y - ray_length * math.sin(math.radians(car_angle+90))

    # Отрисовка луча
    pygame.draw.line(screen, BLACK, (car_x, car_y), (end_x1, end_y1))
    pygame.draw.line(screen, BLACK, (car_x, car_y), (end_x2, end_y2))
    pygame.draw.line(screen, BLACK, (car_x, car_y), (end_x3, end_y3))
    
    for line in barrier_array:
        x1, y1, x2, y2 = line
        pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2))
        if (intersection(x1, y1, x2, y2, car_x, car_y, car_x+10, car_y+10)):
            car_x, car_y = 369, 537
            car_angle = 180 
            print(intersection(x1, y1, x2, y2, car_x, car_y, car_x+10, car_y+10))
        


    # Отрисовка вращенного объекта
    pygame.draw.circle(screen, BLACK, (car_x, car_y), (10))



    line1_distance = find_intersection_points(barrier_array, car_x, car_y, end_x1, end_y1)
    line2_distance = find_intersection_points(barrier_array, car_x, car_y, end_x2, end_y2)
    line3_distance = find_intersection_points(barrier_array, car_x, car_y, end_x3, end_y3)

    distance_line_array = np.array([line1_distance, line2_distance, line3_distance])

    print(distance_line_array)

   
    pygame.display.flip()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car_angle += rotation_speed  # Поворот влево
    if keys[pygame.K_RIGHT]:
        car_angle -= rotation_speed  # Поворот вправо
    # Установка фреймрейта
    clock.tick(60)

pygame.quit()

