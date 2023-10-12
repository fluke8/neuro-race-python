import pygame
import random
import numpy as np
import tensorflow as tf

# Define constants
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 600
BIRD_WIDTH, BIRD_HEIGHT = 40, 40
PIPE_WIDTH, PIPE_HEIGHT = 100, 300
GRAVITY = 0.5
PIPE_SPEED = 5
BIRD_JUMP = -10
NUM_GENERATIONS = 100
NUM_BIRDS_IN_GENERATION = 100

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(2, activation='linear')
])
def crossover_and_mutate(parent1, parent2, mutation_rate=0.01):
    child_weights = []
    
    # Assuming parent1 and parent2 are Keras models
    for layer1, layer2 in zip(parent1.layers, parent2.layers):
        # Crossover (average weights)
        new_weights = (layer1.get_weights()[0] + layer2.get_weights()[0]) / 2.0
        
        # Mutation
        mask = np.random.rand(*new_weights.shape) < mutation_rate
        mutation = np.random.randn(*new_weights.shape) * 0.1  # Adjust the mutation scale as needed
        new_weights = np.where(mask, new_weights + mutation, new_weights)
        
        child_weights.append(new_weights)
    
    # Create a new model with the child weights
    child_model = tf.keras.models.clone_model(parent1)
    child_model.build((None, input_size))  # Assuming input_size is known
    
    for layer, weights in zip(child_model.layers, child_weights):
        layer.set_weights([weights])

    return child_model

def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)
# Define a function to create a new generation of birds
def create_new_generation():
    global model  # Используем глобальную переменную модели

    # Оценка производительности каждой птицы (пример: доли, пройденной дистанции)
    bird_scores = []

    # Задайте вероятности выбора каждой птицы на основе их оценок
    selection_probabilities = softmax(bird_scores)

    # Создайте новое поколение птиц
    new_generation = []

    for _ in range(NUM_BIRDS_IN_GENERATION):
        # Выбор двух родительских птиц с использованием вероятностей
        parent1, parent2 = np.random.choice(NUM_BIRDS_IN_GENERATION, size=2, p=selection_probabilities)

        # Создание потомка птицы (нейронной сети) путем скрещивания и мутации
        child_model = crossover_and_mutate(model[parent1], model[parent2])

        # Добавление потомка в новое поколение
        new_generation.append(child_model)

    # Обновление модели для нового поколения птиц
    model = new_generation

# Main game loop
def main():
    bird_x = SCREEN_WIDTH // 4
    bird_y = SCREEN_HEIGHT // 2
    bird_velocity = 0
    pipes = []

    for generation in range(NUM_GENERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird_velocity = BIRD_JUMP

        # Move bird
        bird_y += bird_velocity
        bird_velocity += GRAVITY

        # Generate pipes
        if len(pipes) == 0 or pipes[-1]["x"] < SCREEN_WIDTH - 200:
            pipe_height = random.randint(100, 400)
            pipes.append({"x": SCREEN_WIDTH, "height": pipe_height})

        # Move pipes
        for pipe in pipes:
            pipe["x"] -= PIPE_SPEED

        # Remove off-screen pipes
        pipes = [pipe for pipe in pipes if pipe["x"] > -PIPE_WIDTH]

        # Check for collisions
        for pipe in pipes:
            if bird_x < pipe["x"] + PIPE_WIDTH and bird_x + BIRD_WIDTH > pipe["x"]:
                if bird_y < pipe["height"] or bird_y + BIRD_HEIGHT > pipe["height"] + PIPE_HEIGHT:
                    bird_x = SCREEN_WIDTH // 4
                    bird_y = SCREEN_HEIGHT // 2
                    bird_velocity = 0
                    pipes = []
                    create_new_generation()
                    break

        # Draw everything
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 128, 0), (bird_x, bird_y, BIRD_WIDTH, BIRD_HEIGHT))
        for pipe in pipes:
            pygame.draw.rect(screen, (0, 0, 0), (pipe["x"], 0, PIPE_WIDTH, pipe["height"]))
            pygame.draw.rect(screen, (0, 0, 0), (pipe["x"], pipe["height"] + PIPE_HEIGHT, PIPE_WIDTH, SCREEN_HEIGHT))
        pygame.display.update()

if __name__ == "__main__":
    main()
