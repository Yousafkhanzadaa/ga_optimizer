import numpy as np
import yfinance as yf
from deap import base, creator, tools, algorithms
import random
import multiprocessing

# Trader provides 'strategy_logic' function and 'param_ranges'
def strategy_logic(dataset, **params):
    # Trader-defined logic here
    # Example: net_profit calculation based on trading signals in dataset using given **params
    net_profit = random.random()  # Placeholder value. Replace with real strategy logic.
    return net_profit,

# Dynamic paraméter setup
param_ranges = {
    'look_back_period': range(5, 50),
    'entry_z_score': (0.5, 3.0),
    'exit_z_score': (0.0, 2.0)
}

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
def evalFitness(individual):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    return strategy_logic(data, **params)
    
toolbox.register("evaluate", evalFitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_optimizer():
    # Parallelize the evaluation
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # GA parameters
    pop_size = 300
    num_generations = 40

    # Create initial population
    pop = toolbox.population(n=pop_size)
    
    # Run GA
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

# Sample usage
ticker_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'

if __name__ == "__main__":
    best_params = run_optimizer()
    print(f"Best strategy parameters: {best_params}")
