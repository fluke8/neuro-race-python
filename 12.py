import neat

# Определение функции оценки (fitness function)
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Создание нейронной сети на основе генома
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Выполнение задачи сети и получение результата
        output = net.activate(input_data)
        
        # Вычисление оценки (fitness) на основе результата
        fitness = calculate_fitness(output)
        
        # Присвоение оценки геному
        genome.fitness = fitness


num_generations = 2

# Создание конфигурации NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config_file.txt')

# Создание популяции
p = neat.Population(config)

# Запуск эволюции
p.run(eval_genomes, num_generations)

