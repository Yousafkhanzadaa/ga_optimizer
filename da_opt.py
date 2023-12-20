from deap import base, creator, tools
import yfinance as yf
import random
import trading_strategy

# This function evaluates a given list of parameters (representing an individual in the GA population) based on the strategy's performance in backtesting:
def evalTradingStrategy(strategy_class, data, individual):
    
    # Map individual genotype to parameter names
    params = {key: value for key, value in zip(strategy_class.parameter_ranges.keys(), individual)}
    # Initialize a new instance of the strategy with the parameters
    strategy_instance = strategy_class(**params)
    # Run backtesting with the given data
    performance = strategy_instance.backtest(data)
    # Return the performance metric (e.g., final net profit)
    return (performance,)


# Create the genetic algorithm primitives in DEAP, using dynamic attribute creation based on the strategy's parameter ranges:
# Custom creator for maximizing the strategy return
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize DEAP's toolbox
toolbox = base.Toolbox()

# Generate attributes for each strategy parameter
def create_attributes(strategy_cls):
    strategy_class = strategy_cls()
    for i, (key, value_range) in enumerate(strategy_class.parameter_ranges.items()):
        # Use 'i' to create unique attribute names: attr_param1, attr_param2, etc.
        toolbox.register(f"attr_{key}", random.uniform, *value_range)

    # Define the individual as a combination of the attributes
    attributes = (getattr(toolbox, f"attr_{key}") for key in strategy_class.parameter_ranges.keys())
    toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)

    # Define the population to be a list of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register genetic operators
def setup_genetic_operators():
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    

# Implement the optimization loop to evolve the population and find the best parameter set:
def optimise_with_ga(strategy_class, data):
    # Create attributes and population based on strategy parameter ranges
    create_attributes(strategy_class)
    population = toolbox.population(n=50)  # Example population size

    # Register the evaluation function with bound arguments for current strategy and data
    toolbox.register("evaluate", evalTradingStrategy, strategy_class, data)

    # Setup genetic operators
    setup_genetic_operators()

    # Example genetic algorithm hyperparameters
    number_of_generations = 40
    crossover_probability = 0.7
    mutation_probability = 0.2

    # Run the genetic optimization algorithm
    for gen in range(number_of_generations):
        # Select and clone the next generation individuals
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population with the next generation
        population[:] = offspring

    # Find and return the best individual and its fitness
    best_ind = tools.selBest(population, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    return best_ind, best_fitness

# Assume 'data' is a Pandas DataFrame containing historical price data including the 'Close' column
data = yf.download(tickers="AAPL", start="2021-01-01", end="2022-01-01")
best_parameters, best_fitness = optimise_with_ga(trading_strategy.MyStrategy, data)