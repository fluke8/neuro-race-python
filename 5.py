import numpy as np

# Определение окружения (сетка 4x4)
# S: Начало, G: Цель, X: Препятствие
env = np.array([['S', 'X', 'O', 'O'],
                ['O', 'X', 'O', 'X'],
                ['O', 'O', 'X', 'X'],
                ['X', 'O', 'O', 'G']])

# Определение действий (Вверх, Вниз, Влево, Вправо)
actions = [0, 1, 2, 3]

# Инициализация Q-таблицы
num_states = env.size
num_actions = len(actions)
Q = np.zeros((num_states, num_actions))


# Гиперпараметры
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Вспомогательная функция для преобразования состояния (i, j) в одиночный индекс
def state_to_index(state):
    print(state[0] * env.shape[1] + state[1])
    print('x')
    return state[0] * env.shape[1] + state[1]

# Вспомогательная функция для выбора действия на основе эпсилон-жадной политики
def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state_to_index(state), :])

import numpy as np

# Инициализация Q-значений
num_states = env.shape[0] * env.shape[1]  # Здесь предполагается, что env - это двумерный массив, представляющий среду
num_actions = 4  # Здесь предполагается, что агент может совершать 4 действия (влево, вверх, вправо, вниз)
Q = np.zeros((num_states, num_actions))

# Гиперпараметры
learning_rate = 0.1
discount_factor = 0.9
num_episodes = 1000  # Количество эпизодов обучения

def state_to_index(state):
    # Функция для преобразования состояния (координат) в индекс состояния в Q-таблице
    return state[0] * env.shape[1] + state[1]

for episode in range(num_episodes):
    state = (0, 0)  # Начальное состояние
    done = False
    while not done:
        action = select_action(state)  # Вам нужно реализовать функцию select_action
        if action == 0:
            next_state = (state[0], state[1]-1)
        elif action == 1:
            next_state = (state[0]+1, state[1])
        elif action == 2:
            next_state = (state[0], state[1]+1)
        elif action == 3:
            next_state = (state[0]-1, state[1])
        reward = 0
        
        if next_state[0] >= 0 and next_state[0] < env.shape[0]-1 and next_state[1] >= 0 and next_state[1] < env.shape[1]-1:
            if env[next_state] == 'G':
                reward = 2
                done = True
            elif env[next_state] == 'X':
                reward = -1
                done = True
            elif env[next_state] == 'C':
                reward = 1
                done = True
        else:
            reward = -2
            done = True
            # Обновление значения Q
        Q[state_to_index(state), action] = (1 - learning_rate) * Q[state_to_index(state), action] + learning_rate * (reward + discount_factor * np.max(Q[state_to_index(next_state), :]))

        state = next_state

        
# Извлечение изученной политики (оптимальных действий)
optimal_policy = np.argmax(Q, axis=1)
print("Оптимальная политика:")
print(optimal_policy.reshape(env.shape))
print(Q)