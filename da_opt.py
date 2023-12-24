import logging
from deap import base, creator, tools
import yfinance as yf
from yfinance import shared
import random
import trading_strategy
import numpy
from multiprocessing import Pool

random.seed(42)
# Create the genetic algorithm primitives in DEAP, using dynamic attribute creation based on the strategy's parameter ranges:
# Custom creator for maximizing the strategy return
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize DEAP's toolbox
toolbox = base.Toolbox()

# Generate attributes for each strategy parameter
def create_attributes():
    strategy_parameter_ranges = trading_strategy.MyStrategy().parameter_ranges
    for param, p_range in strategy_parameter_ranges.items():
        toolbox.register(f"attr_{param}", random.uniform, *p_range)

    # Define the individual as a combination of the attributes
    toolbox.register("individual", tools.initCycle, creator.Individual, (getattr(toolbox, f"attr_{param}") for param in strategy_parameter_ranges), n=1)
    # Define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

# Evaluate trading strategy
def evaluate_individual(individual, strategy_class, data):

    strategy_instance = strategy_class(*individual)
    fitness = strategy_instance.backtest(data)
    return (fitness,)  # Assign a low fitness value

# Register genetic operators
def setup_genetic_operators():
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    

# Implement the optimization loop to evolve the population and find the best parameter set:
def optimise_with_ga(strategy_class, data):
    # Create attributes and population based on strategy parameter ranges
    population = toolbox.population(n=50)  # Example population size
    
    # Setup genetic operators
    setup_genetic_operators()

    # Example genetic algorithm hyperparameters
    number_of_generations = 40
    crossover_probability = 0.7
    mutation_probability = 0.2
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields
    
    # Run the genetic optimization algorithm
    for gen in range(number_of_generations):
        # Select and clone the next generation individuals1
        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        
        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_probability:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_probability:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(lambda ind: evaluate_individual(
            individual=ind, 
            strategy_class=strategy_class, 
            data=data), 
            invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with the next generation
        population[:] = offspring
        create_attributes()
        
        
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        logging.info(logbook.stream)


    # Find and return the best individual and its fitness
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    return best_ind, best_fitness


if __name__ == '__main__':
    create_attributes()
    
    # Main code for optimization
    data = yf.download(tickers="AAPL", start="2021-01-01", end="2022-01-01")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    pool = Pool()
    toolbox.register("map", pool.map)

    # Run the optimization process
    best_parameters, best_fitness = optimise_with_ga(trading_strategy.MyStrategy, data)
    print("Best Parameters:", best_parameters)
    print("Best Fitness:", best_fitness)

    pool.close()
    pool.join()