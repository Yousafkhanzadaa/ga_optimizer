#-------- Implementation of basic GPU-parallel Genetic Algorithm for with python CUDA using Numba  ---------#

import numpy as np
from numba import cuda
import yfinance as yf
import time
import random
import math


#-------- Initial Values For trading Strategy  ---------#
ticker_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2021-01-01'
look_back_period = 20
entry_z_score = 1.0
exit_z_score = 0.0


#-------- Fetch historical data from Yahoo Finance  ---------#
data = yf.download(ticker_symbol, start=start_date, end=end_date)


#-------- Verify CUDA access  ---------#
print(cuda.gpus)


#-------- Mean Reversion Trading Strategy  ---------#
def mean_reversion_strategy(ticker, start_date, end_date, look_back_period, entry_z_score, exit_z_score):
    closing_prices = data['Close'].values

    # Calculate the moving average and standard deviation based on the look-back period
    moving_avg = np.convolve(closing_prices, np.ones(look_back_period)/look_back_period, mode='valid')
    deviation = closing_prices[look_back_period-1:] - moving_avg
    standard_deviation = np.std(deviation)

    # Assuming deviation is normally distributed, z_score for each data point is deviation divided by the standard deviation
    z_scores = deviation / standard_deviation

    # Trading logic
    in_position = False
    entry_price = 0.0
    total_profit = 0.0
    trades = 0

    # Loop through the z_score array
    for i in range(len(z_scores)):
        # Check if not in position and the z_score exceeds entry threshold
        if not in_position and z_scores[i] < -entry_z_score:
            in_position = True
            entry_price = closing_prices[i + look_back_period - 1]
        # Check if in position and the z_score is within exit threshold
        elif in_position and -exit_z_score <= z_scores[i] <= exit_z_score:
            in_position = False
            exit_price = closing_prices[i + look_back_period - 1]
            profit = exit_price - entry_price
            total_profit += profit
            trades += 1

    # Calculate performance
    net_profit = total_profit
    profit_per_trade = total_profit / trades if trades > 0 else 0

    # Returns number of trades, net profit and profit per trade
    return net_profit


#-------- Parallel kernel function using CUDA  ---------#
@cuda.jit
def eval_genomes_kernel(chromosomes, fitnesses, pop_length, chrom_length):
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  # Compute flattened index inside the array
  pos = tx + ty * bw
  if pos < pop_length:  # Check array boundaries
  # in this example the fitness of an individual is computed by an arbitary set of algebraic operations on the chromosome
    net_profite = mean_reversion_strategy(ticker_symbol, start_date, end_date, look_back_period, entry_z_score, exit_z_score)
    fitnesses[pos] = chromosomes[entry_z_score, exit_z_score]
    # Ensure the fitness is non-negative
    if (fitnesses[pos] < 0):
        fitnesses[pos] = 0


#-------- Plain evaluation function, not parallel  ---------#
def eval_genomes_plain(chromosomes, fitnesses):
  for i in range(len(chromosomes)):
    # in this example the fitness of an individual is computed by an arbitary set of algebraic operations on the chromosome
    num_loops = 3000
    for j in range(num_loops):
      fitnesses[i] += chromosomes[i][1] # do the fitness evaluation
    for j in range(num_loops):
      fitnesses[i] -= chromosomes[i][2]
    for j in range(num_loops):
      fitnesses[i] += chromosomes[i][3]

    if (fitnesses[i] < 0):
      fitnesses[i] = 0

#-------- Function to compute next generation in Genetic Algorithm  ---------#
#-------- Performs Selection, Crossover, and Mutation operations  ---------#
def next_generation(chromosomes, fitnesses):
  fitness_pairs = []
  fitnessTotal = 0.0
  for i in range(len(chromosomes)):
    fitness_pairs.append( [chromosomes[i], fitnesses[i]] )
    fitnessTotal += fitnesses[i]

  fitnesses = list(reversed(sorted(fitnesses))) #fitnesses now in descending order
  sorted_pairs = list(reversed(sorted(fitness_pairs, key=lambda x: x[1])))



  new_chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
  #new_brains_fitnesses = []


  #create roulette wheel from relative fitnesses for fitness-proportional selection
  rouletteWheel = []
  fitnessProportions = []
  for i in range(len(chromosomes)):
      fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
      if(i == 0):
          rouletteWheel.append(fitnessProportions[i])
      else:
          rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])

  #Generate new population with children of selected chromosomes

  for i in range(len(chromosomes)):

      #Fitness Proportional Selection
      spin1 = random.uniform(0, 1)      # A random float from 0.0 to 1.0
      spin2 = random.uniform(0, 1)      # A random float from 0.0 to 1.0

      j = 0
      while( rouletteWheel[j] <= spin1 ):
          j += 1

      k = 0
      while( rouletteWheel[k] <= spin2 ):
          k += 1

      genome_copy = sorted_pairs[j][0]    #Genome of parent 1
      genome_copy2 = sorted_pairs[k][0]   #Genome of parent 2

      #create child genome from parents (crossover)
      index = random.randint(0, len(genome_copy) - 1)
      index2 = random.randint(0, len(genome_copy2) - 1)

      child_sequence = []

      for y in range(math.floor(len(genome_copy) / 2)):
          child_sequence.append( genome_copy[ (index + y) % len(genome_copy) ] )

      for y in range(math.floor(len(genome_copy2)/ 2)):
          child_sequence.append( genome_copy2[ (index2 + y) % len(genome_copy2) ] )


      child_genome = np.zeros(len(chromosomes[0]), dtype=np.float32)

      #mutate genome
      for a in range(len(child_sequence)):
        if random.uniform(0,1) < 0.01: # 1% chance of a random mutation
          child_genome[a] = random.uniform(0,1)
        else:
          child_genome[a] = child_sequence[a]

      #Add add new chromosome to next population
      new_chromosomes[i] = child_genome

  #Replace old chromosomes with new
  for i in range(len(chromosomes)):
    for j in range(len(chromosomes[0])):
      chromosomes[i][j] = new_chromosomes[i][j]



#-------- Initialize Population  ---------#
random.seed(1111)
pop_size = 5000
chrom_size = 10
num_generations = 5
fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1) #random float between 0.0 and 1.0

#-------- Measure time to perform some generations of the Genetic Algorithm without CUDA  ---------#

# print("NO CUDA:")
# start = time.time()
# # Genetic Algorithm on CPU
# for i in range(num_generations):
#   print("Gen " + str(i) + "/" + str(num_generations))
#   eval_genomes_plain(chromosomes, fitnesses)
#   next_generation(chromosomes, fitnesses) #Performs selection, mutation, and crossover operations to create new generation
#   fitnesses = np.zeros(pop_size, dtype=np.float32) #Wipe fitnesses


# end = time.time()
# print("time elapsed: " + str((end-start)))
# print("First chromosome: " + str(chromosomes[0])) #To show computations were the same between both tests


#-------- Prepare kernel ---------#
# Set block & thread size
threads_per_block = 256
blocks_per_grid = (chromosomes.size + (threads_per_block - 1))

#--------- Initialize population again for a new run -------------- #
random.seed(1111)
fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1) #random float between 0.0 and 1.0

#-------- Measure time to perform some generations of the Genetic Algorithm with CUDA  ---------#
print("CUDA:")
start = time.time()
# Genetic Algorithm on GPU
for i in range(num_generations):
  print("Gen " + str(i) + "/" + str(num_generations))
  chromosomes_flat = chromosomes.flatten()

  eval_genomes_kernel[blocks_per_grid, threads_per_block](chromosomes_flat, fitnesses, pop_size, chrom_size)
  next_generation(chromosomes, fitnesses) #Performs selection, mutation, and crossover operations to create new generation
  fitnesses = np.zeros(pop_size, dtype=np.float32) # Wipe fitnesses


end = time.time()
print("time elapsed: " + str((end-start)))
print("First chromosome: " + str(chromosomes[0])) #To show computations were the same between both tests
#-------------------------------------------------------#