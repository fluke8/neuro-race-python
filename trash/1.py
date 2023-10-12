import pygame
import math
import numpy as np
from pygame.math import Vector2

def distance(x1, y1, x2, y2):
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
        
        if (min(x1, x2) <= intersection_point.x <= max(x1, x2) and
            min(y1, y2) <= intersection_point.y <= max(y1, y2) and
            min(x3, x4) <= intersection_point.x <= max(x3, x4) and
            min(y3, y4) <= intersection_point.y <= max(y3, y4)):
            return intersection_point
            
    return None

def find_intersection_points(barrier_array, car_x, car_y, end_x, end_y):
    intersection_points = []
    for i, barrier in enumerate(barrier_array):
        intersection_point = intersection(barrier[0], barrier[1], barrier[2], barrier[3], car_x, car_y, end_x, end_y)
        if intersection_point:
            intersection_points.append(intersection_point)
    return intersection_points

def find_distance_to_intersection(intersection_points, car_x, car_y):
    closest_distance = float('inf')
    closest_intersection = None

    for point in intersection_points:
        x1, y1 = point
        # Рассчитываем расстояние от автомобиля до точки пересечения
        
        dist = math.sqrt((x1 - car_x)**2 + (y1 - car_y)**2)

        if dist < closest_distance:
            closest_distance = dist
            closest_intersection = point

    return closest_distance, closest_intersection

pygame.init()

WINDOW_WIDTH, WINDOW_HEIGHT = 1268, 840

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Движение по трассе")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

car_width = 20
car_height = 20
car_color = BLACK
car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
pygame.draw.circle(car_surface, car_color, (20,20), car_width)

car_x = 369
car_y = 537

speed = 2   
rotation_speed = 3  
car_angle = 180  

barrier_array = np.array([[20, 378, 64, 130], [64, 130, 445, 54], [445, 54, 856, 116], [856, 116, 1157, 307], [1157, 307, 1040, 649], [1040, 649, 689, 714], [689, 714, 460, 680], 
                          [460, 680, 187, 576], [20, 378, 187, 576],
                          [159, 344, 189, 200],[189, 200, 443, 150],[443, 150, 820, 234],[820, 234, 969, 345],[969, 345, 959, 500],[959, 500, 875, 556],[875, 556, 582, 560],
                          [582, 560, 159, 344]])  

reward_lines_array = np.array([[187, 576, 304, 415], [20, 378, 159, 344], [64, 130, 189, 200], [445, 54, 443, 150],
                                [856, 116,820, 234], [1157, 307,969, 345], [1040, 649, 875, 556], [959, 500, 1079, 543], [582, 560, 550, 696]])

num_rays = 8
ray_length = 5000


clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE) 

    radians = math.radians(car_angle)
    position = Vector2(car_x, car_y)
    direction = Vector2(math.cos(radians), -math.sin(radians))
    position += direction * speed
    car_x, car_y = position.x, position.y



    end_xy = np.zeros((num_rays, 2))


    for i in range(num_rays):
        for j in range(2):
            if j==1:
                end_xy[i][j] = car_y - ray_length * math.sin(math.radians(car_angle+i*360/num_rays))
            else:
                end_xy[i][j] = car_x + ray_length * math.cos(math.radians(car_angle+i*360/num_rays))
    #print(car_angle)


    
    for line in barrier_array:
        x1, y1, x2, y2 = line
        # pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2))
        if (intersection(x1, y1, x2, y2, car_x-10, car_y-10, car_x+10, car_y+10)):
            car_x, car_y = 369, 537
            car_angle = 180 
            print(intersection(x1, y1, x2, y2, car_x, car_y, car_x+10, car_y+10))

    for line in reward_lines_array:
        x1, y1, x2, y2 = line
        #pygame.draw.line(screen, GREEN, (x1, y1), (x2, y2))


    pygame.draw.circle(screen, BLACK, (car_x, car_y), (10))

    line_intersection_xy = []
    for end_x,end_y in end_xy:
       line_intersection_xy.append(find_intersection_points(barrier_array, car_x, car_y, end_x, end_y))


    for i, intersection_points in enumerate(line_intersection_xy):
        closest_distance, closest_intersection = find_distance_to_intersection(intersection_points, car_x, car_y)
        if closest_intersection:
            pygame.draw.circle(screen, RED, (int(closest_intersection[0]), int(closest_intersection[1])), 5)
            if i == 0:
                pygame.draw.line(screen, BLACK, (car_x, car_y), (int(closest_intersection[0]), int(closest_intersection[1])), 5)
            else:
                pygame.draw.line(screen, BLACK, (car_x, car_y), (int(closest_intersection[0]), int(closest_intersection[1])), 1)


    
    distance_line_array = []
    for i in range(len(line_intersection_xy)):
        distance, closest_intersection = find_distance_to_intersection(line_intersection_xy[i], car_x, car_y)
        if closest_intersection is not None and not math.isinf(distance):
            distance_line_array.append(int(distance))
        else:
            # Обработка случая, когда расстояние бесконечно или ближайшее пересечение отсутствует
            distance_line_array.append(None)



    print(distance_line_array)

    # for i, (end_x, end_y) in enumerate(end_xy):
    #     if i == 0:
    #         # Ваш код для первой итерации
    #         pygame.draw.line(screen, BLACK, (car_x, car_y), (end_x, end_y), 3)
    #     else:
    #         # Ваш код для остальных итераций
    #         pygame.draw.line(screen, BLACK, (car_x, car_y), (end_x, end_y), 1)
    

    pygame.display.flip()
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        car_angle += rotation_speed  
    if keys[pygame.K_RIGHT]:
        car_angle -= rotation_speed  



    clock.tick(60)
pygame.quit()
