import numpy as np
import yfinance as yf
from deap import base, creator, tools, algorithms
import random
import multiprocessing
from numba import cuda

# Sample usage
ticker_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'


# Dynamic paraméter setup
param_ranges = {
    'look_back_period': range(5, 50),
    'entry_z_score': (0.5, 3.0),
    'exit_z_score': (0.0, 2.0)
}

def strategy_logic(dataset, **params):
    # Trader-defined logic here
    # Example: net_profit calculation based on trading signals in dataset using given **params
    net_profit = random.random()  # Placeholder value. Replace with real strategy logic.
    return net_profit,

@cuda.jit(device=True)
def device_strategy_logic(individual_params):
    # Replace with real CUDA compatible strategy logic
    result = 0.0
    for param in individual_params:
        result += param  # Simplified example calculation
    return result

@cuda.jit
def gpu_eval_fitness(population_params, fitnesses):
    # Kernel evaluates fitness for the whole population, not one individual
    idx = cuda.grid(1)
    if idx < fitnesses.size:
        fitnesses[idx] = device_strategy_logic(population_params[idx])


# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator for parameters
def uniform_random(attr_range):
    return random.uniform(*attr_range) if isinstance(attr_range, tuple) else random.choice(attr_range)

# Structure initializers
toolbox.register("attr_float", uniform_random, attr_range=(0.0, 1.0)) # Placeholder range
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [uniform_random(param_ranges[k]) for k in param_ranges.keys()])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def gpu_wrapper(individual):
    # Convert the individual's parameters for GPU processing
    
    # Prepare individuals matrix and output array for single individual
    individual_matrix = np.array(individual, dtype=np.float32).reshape(1, -1)
    fitnesses_host = np.zeros(1, dtype=np.float32)

    # Copy data to GPU
    d_individual = cuda.to_device(individual_matrix)
    d_fitnesses = cuda.to_device(fitnesses_host)
    
    # Launch kernel for a single individual
    threads_per_block = 256
    gpu_eval_fitness[1, threads_per_block](d_individual, d_fitnesses)

    # Copy the fitness back
    cuda.synchronize()
    fitnesses_host = d_fitnesses.copy_to_host()

    # Copy fitness back and return as tuple
    fitnesses = d_fitnesses.copy_to_host()
    return fitnesses[0],

toolbox.register("evaluate", gpu_wrapper)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_optimizer():
    # Fetch dataset from yfinance
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # GA parameters
    pop_size = 300
    num_generations = 5
    cxpb = 0.5  # Crossover probability
    mutpb = 0.2

    # Initialize population
    pop = toolbox.population(n=pop_size)

    # CUDA-related preparations - Prepare the whole population matrix and fitness array
    individuals_matrix = np.array([ind[:] for ind in pop]).astype(np.float32)
    fitnesses_host = np.zeros(len(pop), dtype=np.float32)

    # Copying the whole population to the GPU at once instead of individual copying
    d_individuals = cuda.to_device(individuals_matrix)
    d_fitnesses = cuda.to_device(fitnesses_host)

    # Set threads and blocks properly
    threads_per_block = 256
    blocks = (len(pop) + (threads_per_block - 1)) // threads_per_block

    # GA run - proper synchronization of CUDA operations
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    
    for gen in range(num_generations):
          # Evaluate the entire population
          gpu_eval_fitness[blocks, threads_per_block](d_individuals, d_fitnesses)
          
          # Copy fitnesses back after kernel execution
          cuda.synchronize()
          fitnesses_host = d_fitnesses.copy_to_host()

          # Update fitnesses in the population
          for ind, fit in zip(pop, fitnesses_host):
              ind.fitness.values = fit,

          # Select the next generation individuals
          offspring = toolbox.select(pop, len(pop))
          # Clone the selected individuals
          offspring = list(map(toolbox.clone, offspring))

          # Apply crossover and mutation on the offspring
          for child1, child2 in zip(offspring[::2], offspring[1::2]):
              if random.random() < cxpb:
                  toolbox.mate(child1, child2)
                  del child1.fitness.values
                  del child2.fitness.values

          for mutation in offspring:
              if random.random() < mutpb:
                  toolbox.mutate(mutation)
                  del mutation.fitness.values

          # Evaluate the individuals with an invalid fitness
          invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
          fitnesses = map(toolbox.evaluate, [ind[:] for ind in invalid_ind])
          for ind, fit in zip(invalid_ind, fitnesses):
              ind.fitness.values = fit

          # The population is entirely replaced by the offspring
          pop[:] = offspring

          # Gather all the fitnesses in one list and print the stats
          record = stats.compile(pop)
          hof.update(pop)
          print(record)

    # Extract info about the best individual
    best_ind = hof.items[0]
    return best_ind, best_ind.fitness.values

if __name__ == "__main__":
    best_params, best_fitness = run_optimizer()
    print(f"Best strategy parameters: {best_params}")
    print(f"With fitness: {best_fitness}")
