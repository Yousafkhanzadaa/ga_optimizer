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
def device_strategy_logic(individual, params):
    # Simulated device-compatible strategy logic
    result = 0.0
    for value, param in zip(individual, params):
        result += value * param  # Simplified example calculation
    return result

@cuda.jit
def gpu_eval_fitness(population, params, fitnesses):
    # CUDA kernel to evaluate fitnesses
    idx = cuda.grid(1)

    # Check if idx is within bounds of the population
    if idx < population.shape[0]:
        # Compute fitness for an individual
        # In reality, the strategy logic would be more complex
        fitnesses[idx] = device_strategy_logic(population[idx], params)


# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
def uniform_random(attr_range):
    if isinstance(attr_range, range): return random.choice(attr_range)
    return random.uniform(*attr_range)

# Structure initializers
toolbox.register("attr_float", uniform_random, attr_range=(0.0, 1.0)) # Placeholder range
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_float,), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the fitness evaluation function
# def evalFitness(individual):
#     params = {k: uniform_random(param_ranges[k]) for k in param_ranges.keys()}
#     return strategy_logic(data, **params)
    
def gpu_wrapper(individual):
    # Convert the individual's parameters for GPU processing
    parameter_values = np.array([uniform_random(param_ranges[k]) for k in param_ranges.keys()], dtype=np.float32)
    
    # Prepare individuals matrix and output array for single individual
    individual_matrix = np.array(individual, dtype=np.float32).reshape(1, -1)
    fitnesses_host = np.zeros(1, dtype=np.float32)

    # Copy data to GPU
    d_individual = cuda.to_device(individual_matrix)
    d_params = cuda.to_device(parameter_values)
    d_fitnesses = cuda.to_device(fitnesses_host)
    
    # Launch the kernel for a single individual
    gpu_eval_fitness[1, 1](d_individual, d_params, d_fitnesses)

    # Copy the fitness back
    cuda.synchronize()
    fitnesses_host = d_fitnesses.copy_to_host()

    # Return the fitness
    return fitnesses_host[0],

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
    cuda_params = np.array([1.0, 0.5, 0.2])
    
    # params = {k: uniform_random(param_ranges[k]) for k in param_ranges.keys()}
    
    # population_matrix = np.array([ind[:] for ind in pop]).astype(np.float32)
    
    # fitnesses_host = np.zeros(population_matrix.shape[0], dtype=np.float32)
    
     # Copy data to GPU
    # d_population = cuda.to_device(population_matrix)
    # d_params = cuda.to_device(cuda_params)
    # d_fitnesses = cuda.to_device(fitnesses_host)
    
    # Set up blocks and grid for the kernel launch
    # threads_per_block = 64
    # blocks_per_grid = (population_matrix.shape[0] + (threads_per_block - 1)) // threads_per_block
    
    # Run the fitness evaluation on GPU
    # gpu_eval_fitness[blocks_per_grid, threads_per_block](d_population, d_params, d_fitnesses)

    # Wait for all threads to finish
    # cuda.synchronize()

    # # Copy the fitness results back to the host
    # fitnesses_host[:] = d_fitnesses.copy_to_host()

    # Update the fitness scores in the individuals (population)
    # for ind, fitness_value in zip(pop, fitnesses_host):
    #     ind.fitness.values = (fitness_value,)
    
    # Run GA
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

if __name__ == "__main__":
    best_params = run_optimizer()
    print(f"Best strategy parameters: {best_params}")
