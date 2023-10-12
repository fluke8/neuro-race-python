import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
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
from sklearn.preprocessing import StandardScaler
import pygame

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

class MyCarEnv(gym.Env):
    def __init__(self):
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
        self.speed = 8
        self.rotation_speed = 5

        self.num_rays = 12
        self.ray_length = 1000

        
        self.WINDOW_WIDTH, self.WINDOW_HEIGHT = 1268, 840

        self.car_color = self.BLACK
        self.car_width = 20
        self.car_height = 20

        self.last_reward_line = 4

        self.state = np.ones(self.num_rays, dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_rays,))

        self.reward = 0
        self.car_surface = pygame.Surface((self.car_width, self.car_height), pygame.SRCALPHA)
        
        self.distance_array_for_fit = np.ones((1, self.num_rays), dtype=np.float32)


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
                self.reward += -5
                done = True

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
                distance_line_array.append(distance)
            else:
                distance_line_array.append(1000)
                # self.reward = -10
                # done = True
        
        # self.car_xy = np.array([int(self.car_x), int(self.car_y)])

        # self.state = np.concatenate([self.car_xy, distance_line_array])
        distance_line_array = np.array(distance_line_array)
        self.distance_array_for_fit = np.vstack((distance_line_array, self.distance_array_for_fit))[:500]
   

        scaler.fit(self.distance_array_for_fit)
        scaled_distance_line_array = scaler.transform(distance_line_array.reshape(1, -1))

        self.state = scaled_distance_line_array.flatten()
        
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
        
        clock.tick(120)

env = MyCarEnv()



# Определение архитектуры нейронной сети для политики
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1) # Функция softmax для получения вероятностей действий
        return x

# Функция для сэмплирования действия на основе политики
def select_action(policy_net, state):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy_net(state)
    action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
    return action


# Обучение агента
def train(policy_net, optimizer, episodes):
    max_reward = -10
    for episode in range(episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []

        while True:
            action = select_action(policy_net, state)
            next_state, reward, done = env.step(action)

            if render:
                env.render()

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            if done:
                break

            state = next_state

        # Обновление политики на основе Policy Gradient
        returns = []
        R = 0
        for r in episode_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)

        policy_loss = []
        for i in range(len(episode_states)):
            action = episode_actions[i]
            G = returns[i]
            state = episode_states[i]
            action_prob = policy_net(torch.tensor(state, dtype=torch.float32))[action]
            policy_loss.append(-torch.log(action_prob) * G)

        optimizer.zero_grad()
        # Filter out zero-dimensional tensors
        policy_loss = torch.stack(policy_loss).sum()

        policy_loss.backward()
        optimizer.step()
        if sum(episode_rewards) > max_reward:
            max_reward = sum(episode_rewards)
        print(f"Episode {episode}: Total Reward = {sum(episode_rewards)} Max Reward = {max_reward}")
        if episode%1000 == 0:
            torch.save(policy_net.state_dict() , 'policy_net.pth')



def test(policy_net, num_episodes=10):
    total_reward = -10
    for episode in range(num_episodes):
        
        state = env.reset()

        while True:
            action = select_action(policy_net, state)
            next_state, reward, done = env.step(action)

            total_reward += reward

            if render:
                env.render()

            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break

            state = next_state

    env.close()
    return total_reward

# Гиперпараметры
input_size = env.num_rays
output_size = 3
hidden_size1 = 20
learning_rate = 0.01
gamma = 0.01 # Дисконтированный фактор наград
scaler = StandardScaler()

render = False

if render:
    env.screen = pygame.display.set_mode((env.WINDOW_WIDTH, env.WINDOW_HEIGHT))

device = torch.device("cpu")

policy_net = PolicyNetwork(input_size, hidden_size1, output_size)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# path = 'policy_net.pth'

# policy_net.load_state_dict(torch.load(path))
# Обучение агента
episodes = 999999999999
train(policy_net, optimizer, episodes)


