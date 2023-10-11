import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
from gym import spaces
import math
from pygame.math import Vector2
import pygame

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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
        self.start_x = 369
        self.start_y = 537
        self.start_angle = 180 

        self.car_x = self.start_x 
        self.car_y = self.start_y
        self.car_angle = self.start_angle
        self.speed = 1
        self.rotation_speed = 5

        self.num_rays = 8
        self.ray_length = 10000

        
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = 1268, 840

        self.car_color = self.BLACK
        self.car_width = 20
        self.car_height = 20

        self.num_steps = 0

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
        return self.state

    def step(self, action):   
        done = False
        self.reward = 0
        self.num_steps += 1
        if action==0:
            self.car_angle += self.rotation_speed  
        elif action==2:
            self.car_angle -= self.rotation_speed 

        for line in self.reward_lines_array:
            x1, y1, x2, y2 = line
            if (intersection(x1, y1, x2, y2, self.car_x-5, self.car_y-5, self.car_x+5, self. car_y+5)):

                self.reward += 1

        for line in self.barrier_array:
            x1, y1, x2, y2 = line
            if (intersection(x1, y1, x2, y2, self.car_x-10, self.car_y-10, self.car_x+10, self. car_y+10)):
                self.reward = -10
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
                distance_line_array.append((distance))
            else:
                distance_line_array.append(9999999)
                # self.reward = -10
                # done = True

        self.state = distance_line_array 

        
        return self.state, self.num_steps, done

    def render(self, mode='human'):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.draw.circle(self.car_surface, self.car_color, (20,20), self.car_width)

        clock = pygame.time.Clock()
        self.screen.fill(self.WHITE) 


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
        
        clock.tick(120)
        
        


# Создание экземпляра вашей среды
env = MyCarEnv()

# Определение архитектуры нейронной сети
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Гиперпараметры
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
num_episodes = 10000
max_steps_per_episode = 1000

# Создание среды
env = MyCarEnv()

# Инициализация Q-сети
input_size = env.observation_space.shape[0]  # Размер входного состояния
output_size = env.action_space.n  # Количество возможных действий
q_network = QNetwork(input_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
# model_save_path = 'q_network.pth'
# q_network.load_state_dict(torch.load(model_save_path))
#q_network.eval() 

log_file = open('training_log.txt', 'w')


# Move your data (input tensors) to the GPU


# Цикл обучения
episode = 0

env.screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
while True:
    episode +=1

    state = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Выбор действия с использованием эпсилон-жадной стратегии
        if random.random() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()

        # Выполнение действия в среде и наблюдение за новым состоянием и наградой
        next_state, reward, done = env.step(action)
        env.render()

                
        # print(next_state, reward, done)
        # Обновление Q-значения на основе беллмановского уравнения
        with torch.no_grad():
            q_next = q_network(torch.FloatTensor(next_state))
            max_q_next, _ = torch.max(q_next, dim=0)
            target_q = reward + gamma * max_q_next

        q_values = q_network(torch.FloatTensor(state))
        q_value = q_values[action]

        loss = nn.MSELoss()(q_value, target_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        state = next_state

        if done:
            break
    # torch.save(q_network.state_dict(), model_save_path)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    with open('training_log.txt', 'a') as log_file:
        log_file.write(f"Episode {episode + 1}: Total Reward = {total_reward}\n")


# Использование обученной сети для управления автомобилем

env.screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))
state = env.reset()
pygame.init()
pygame.display.set_caption("Движение по трассе")


for _ in range(max_steps_per_episode):
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state)
        q_values = q_network(state_tensor)
        action = q_values.argmax().item()

    next_state, _, done = env.step(action)
    env.render()

    state = next_state

    if done:
        break

env.close()
