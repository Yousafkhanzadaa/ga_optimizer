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
toolbox.register("individual", tools.initCycle, creator.Individual, (lambda: [uniform_random(param_ranges[k]) for k in param_ranges.keys()]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=300)

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
    gpu_eval_fitness[1, threads_per_block](d_individual_params, d_fitnesses)

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
    
    # Parallelize the evaluation
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # GA parameters
    pop_size = 300
    num_generations = 40

    # Create initial population
    pop = toolbox.population(n=pop_size)
    # cuda_params = np.array([1.0, 0.5, 0.2])
    
    # Run GA
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the algorithm using simple EA
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)
    
    # Extract the best individual's parameters and their corresponding fitness
    best_ind = hof.items[0]
    return best_ind, best_ind.fitness.values

if __name__ == "__main__":
    best_params, best_fitness = run_optimizer()
    print(f"Best strategy parameters: {best_params}")
    print(f"With fitness: {best_fitness}")
