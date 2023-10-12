import numpy as np
import gym

# Создание среды "FrozenLake"
env = gym.make("FrozenLake-v1")

# Инициализация Q-таблицы
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Параметры обучения
learning_rate = 0.1
discount_factor = 0.99
num_episodes = 1000

# Обучение Q-таблицы
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Выбор действия с вероятностью epsilon-greedy
        epsilon = 0.1
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Случайное действие
        else:
            action = np.argmax(Q[state, :])  # Выбор действия с максимальным Q-значением

        # Выполнение действия и получение нового состояния, вознаграждения и флага завершения
        new_state, reward, done, _ = env.step(action)

        # Обновление Q-значения
        Q[state, action] = (1 - learning_rate) * Q[state, action] + \
                           learning_rate * (reward + discount_factor * np.max(Q[new_state, :]))

        state = new_state

# Тестирование обученной модели
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[state, :])
    new_state, _, done, _ = env.step(action)
    env.render()
    state = new_state
