import gym
import numpy as np
import gym
from gym import spaces
import math
from pygame.math import Vector2
import pygame
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

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

def is_point_inside_barriers(x, y, barriers):
    num_intersections = 0
    
    for barrier in (barriers):
        x1, y1, x2, y2 = barrier
        
        # Проверяем, пересекается ли луч с барьером
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            num_intersections += 1
    
    # Если число пересечений нечетное, точка находится внутри зоны
    return num_intersections % 2 == 1


def is_point_outside_barriers(x,y, barriers):
    for barrier in barriers:
        x1, y1, x2, y2 = barrier
        # Проверяем, лежит ли точка (x, y) между двумя точками линии
        if not(x1 <= x <= x2 or x2 <= x <= x1) and not(y1 <= y <= y2 or y2 <= y <= y1):
            print(-1)
            return True
    return False

def is_point_inside_rectangle(x, y, rectangle):

    x1, y1 = rectangle[0]
    x2, y2 = rectangle[1]
    x3, y3 = rectangle[2]
    x4, y4 = rectangle[3]

    # Проверяем, лежит ли точка внутри прямоугольника с помощью условий для x и y
    is_inside_x = x1 <= x <= x3 
    is_inside_y = y3 <= y <= y1 
    return is_inside_x and is_inside_y

class MyCarEnv(gym.Env):
    def __init__(self):
        # Определение пространства действий (предположим, что у вас есть два действия: вперед и назад)
        self.action_space = spaces.Discrete(3)        

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)

        self.barrier_array_outside = np.array([[20, 378, 64, 130], [64, 130, 445, 54], [445, 54, 856, 116], [856, 116, 1157, 307], [1157, 307, 1040, 649], [1040, 649, 689, 714], [689, 714, 460, 680], 
                                [460, 680, 187, 576], [20, 378, 187, 576]])

        self.barrier_array_inside = np.array([[159, 344, 189, 200],[189, 200, 443, 150],[443, 150, 820, 234],[820, 234, 969, 345],[969, 345, 959, 500],[959, 500, 875, 556],[875, 556, 582, 560],
                                [582, 560, 159, 344]])  
        
        self.barrier_array = np.concatenate((self.barrier_array_outside, self.barrier_array_inside))

        self.reward_lines_array = np.array([[187, 576, 304, 415], [20, 378, 159, 344], [64, 130, 189, 200], [445, 54, 443, 150],
                                        [856, 116,820, 234], [1157, 307,969, 345], [1040, 649, 875, 556], [959, 500, 1079, 543], [582, 560, 550, 696]])
        
        self.rect = np.array([[17,725],[17,57],[1157, 57], [1157, 725]])
        self.start_x = 369
        self.start_y = 537
        self.start_angle = 180 

        self.car_x = self.start_x 
        self.car_y = self.start_y
        self.car_angle = self.start_angle
        self.speed = 10
        self.rotation_speed = 5

        self.num_rays = 12
        self.ray_length = 10000

        
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = 1268, 840

        self.car_color = self.BLACK
        self.car_width = 20
        self.car_height = 20

        self.last_reward_line = 4

        self.state = np.ones(self.num_rays, dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_rays,))

        self.reward = 0
        self.car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        

    def reset(self):

        self.car_x = self.start_x 
        self.car_y = self.start_y
        self.car_angle = self.start_angle
        self.num_steps = 0
        self.reward = 0
        self.last_reward_line = 4
        return self.state

    def step(self, action):   
        done = False

        self.reward = 0
        self.num_steps += 1
        if action==0:
            self.car_angle += self.rotation_speed  
        elif action==2:
            self.car_angle -= self.rotation_speed 

        for index, line in enumerate(self.reward_lines_array):
            x1, y1, x2, y2 = line
            if self.last_reward_line != index and ((intersection(x1, y1, x2, y2, self.car_x-15, self.car_y-15, self.car_x+15, self.car_y+15)) or 
                                                   (intersection(x1, y1, x2, y2, self.car_x-15, self.car_y+15, self.car_x+15, self.car_y-15))):
                self.reward = 1
                self.last_reward_line = index

        for line in self.barrier_array:
            x1, y1, x2, y2 = line
            if (intersection(x1, y1, x2, y2, self.car_x-15, self.car_y-15, self.car_x+15, self.car_y+15)) or (intersection(x1, y1, x2, y2, self.car_x-15, self.car_y+15, self.car_x+15, self.car_y-15)):
                self.reward += -1
                done = True
        
        if not(is_point_inside_rectangle(self.car_x, self.car_y, self.rect)):
            self.reward += -1
            done = True

        # if is_point_inside_barriers(self.car_x, self.car_y, self.barrier_array_inside):
        #     reward += -10
        #     done = True



        radians = math.radians(self.car_angle)
        position = Vector2(self.car_x, self.car_y)
        direction = Vector2(math.cos(radians), -math.sin(radians))
        position += direction * self.speed
        self.car_x, self.car_y = position.x, position.y

        end_xy = np.zeros((self.num_rays, 2))


        for i in range(self.num_rays):
            for j in range(2):
                if j==1:
                    end_xy[i][j] = self.car_y - self.ray_length * math.sin(math.radians( self.car_angle+i*360/ self.num_rays))
                else:
                    end_xy[i][j] = self.car_x + self.ray_length * math.cos(math.radians( self.car_angle+i*360/ self.num_rays))

        self.line_intersection_xy = []
        for end_x,end_y in end_xy:
            self.line_intersection_xy.append(find_intersection_points( self.barrier_array, self.car_x, self.car_y, end_x, end_y))

        distance_line_array = []
        for i in range(len(self.line_intersection_xy)):
            distance, closest_intersection = find_distance_to_intersection(self.line_intersection_xy[i], self.car_x, self.car_y)
            if closest_intersection is not None and not math.isinf(distance):
                distance_line_array.append(int((distance)/100))
            else:
                distance_line_array.append(9999999)
                # self.reward = -10
                # done = True

        self.state = distance_line_array 

        
        return self.state, self.reward, done

    def render(self, mode='human'):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.draw.circle(self.car_surface, self.car_color, (20,20), self.car_width)

        clock = pygame.time.Clock()
        self.screen.fill(self.WHITE) 

        pygame.draw.line(self.screen, self.BLACK, (self.car_x-15, self.car_y-15), (self.car_x+15, self.car_y+15), 5)
        pygame.draw.line(self.screen, self.BLACK, (self.car_x-15, self.car_y+15), (self.car_x+15, self.car_y-15), 5)
        for line in self.barrier_array:
            x1, y1, x2, y2 = line
            pygame.draw.line(self.screen, self.BLACK, (x1, y1), (x2, y2))


        for line in self.reward_lines_array:
            x1, y1, x2, y2 = line
            pygame.draw.line(self.screen, self.GREEN, (x1, y1), (x2, y2))


        pygame.draw.circle(self.screen, self.BLACK, (self.car_x, self.car_y), (10))



        for i, intersection_points in enumerate(self.line_intersection_xy):
            closest_distance, closest_intersection = find_distance_to_intersection(intersection_points, self.car_x, self.car_y)
            if closest_intersection:
                if i == 0:
                    pygame.draw.line(self.screen, self.BLACK, (self.car_x, self.car_y), (int(closest_intersection[0]), int(closest_intersection[1])), 5)
                else:
                    pygame.draw.line(self.screen, self.BLACK, (self.car_x, self.car_y), (int(closest_intersection[0]), int(closest_intersection[1])), 1)
                pygame.draw.circle(self.screen, self.RED, (int(closest_intersection[0]), int(closest_intersection[1])), 5)
                
        

        pygame.display.flip()


        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.car_angle += self.rotation_speed  
        if keys[pygame.K_RIGHT]:
            self.car_angle -= self.rotation_speed  
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
        
        clock.tick(30)
        
        


# Создание экземпляра вашей среды
env = MyCarEnv()

# Нейронная сеть для аппроксимации Q-функции
model = keras.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(env.action_space.n, activation='linear')
])

# Оптимизатор и функция потерь
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true))


# Гиперпараметры обучения
gamma = 0.95  # Дисконтный фактор
epsilon = 0.1  # Эпсилон-жадная стратегия для исследования
num_episodes = 9999999999
env.screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))

for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        # Выбор действия с использованием эпсилон-жадной стратегии
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            q_values = model.predict(np.array(state).reshape(1, -1))
            action = np.argmax(q_values)  # Выбор действия с наибольшей оценкой

        # Выполнение действия в среде и получение нового состояния и награды
        print(action)
        next_state, reward, done = env.step(action)
        env.render()

        # Вычисление целевого значения Q-функции
        q_values_next = model.predict(np.array(next_state).reshape(1, -1))
        target = reward + gamma * np.max(q_values_next)

        # Обновление Q-функции с использованием обратного распространения ошибки
        with tf.GradientTape() as tape:
            q_values = model(np.array(state).reshape(1, -1))
            loss = loss_fn(target, q_values[0][action])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        episode_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    if episode % 1000 == 0:
        model.save('cartpole_model1.h5')

# Сохранение обученной модели
