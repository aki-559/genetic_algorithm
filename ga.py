import numpy as np
from numpy.random import choice, shuffle
from scipy.spatial.distance import cdist

from plot import plot_function2d, plot_cities

CODE_GRAY = 0
CODE_BINARY = 1

def roulette_wheel(x, fitness, n_samples):
  fitness -= fitness.min() - 1e-5
  prob = fitness / fitness.sum()
  index = choice(len(x), size=n_samples, p=prob)

  return x[index]

def tournament(x, fitness, n_samples, size=5):
  index = choice(len(x), size=(n_samples, size))
  argmax = np.argmax(fitness[index], axis=1)
  index = index[np.arange(n_samples), argmax]

  return x[index]

def elite(x, fitness, n_samples):
  top = np.argpartition(-fitness, n_samples)[:n_samples]
  fitness = fitness[top]
  index = np.arange(len(x))[top]
  index = index[np.argsort(-fitness)]

  return x[index]

TYPE_UNIFORM = 0
TYPE_ONE = 1
TYPE_TWO = 2

def crossover(x, prob, mode):
  shuffle(x)
  p1, p2 = np.vsplit(x, 2)
  index = choice(2, size=len(p1), p=[1-prob, prob]).astype(bool)
  length = x.shape[-1]
  
  not_ = np.logical_not(index)
  new = np.vstack((p1[not_], p2[not_]))
  for x1, x2 in zip(p1[index], p2[index]):
    if mode == TYPE_UNIFORM:
      p = np.random.rand()
      index = choice(2, size=len(x1), p=[1-p, p]).astype(bool)
      tmp = x1[index]
      x1[index], x2[index] = x2[index], tmp
    else:
      if mode == TYPE_ONE:
        start = choice(np.arange(1, length))
        end = length
      elif mode == TYPE_TWO:
        start = choice(np.arange(1, length))
        end = choice(np.arange(start+1, length+1))

      tmp = x1[start:end].copy()
      x1[start:end], x2[start:end] = x2[start:end], tmp
      
    new = np.append(new, np.vstack((x1, x2)), axis=0)

  return new

def mutate(x, prob, is_graph=False):
  index = choice(2, size=len(x), p=[1-prob, prob]).astype(bool)
  new = x[np.logical_not(index)]
  x = x[index]
  count, length = x.shape

  pos = choice(length, size=count)
  x[np.arange(count), pos] = np.random.randint(100) % (length - pos) if is_graph else np.logical_not(x[np.arange(count), pos])

  return np.vstack((new, x))

def gray_to_binary(x):
  return np.add.accumulate(x[:,::-1], axis=1)[:,::-1] % 2

def binary_to_gray(x):
  last = np.zeros(len(x))[:,np.newaxis]
  slided = np.hstack((x[:,1:], last))
  return np.logical_xor(x, slided).astype(float)

def ptype(binary, min_, max_):
  length = binary.shape[-1]
  powered = np.exp2(np.arange(length))
  r = np.sum(binary * powered, axis=1) / (2 ** length)

  return (max_ - min_) * r + min_

def simple_repr(index):
  index = index.copy()
  for i in range(2, index.shape[-1]+1):
    j = index.shape[-1]-i
    tmp = index[:,j+1:]
    bound = np.repeat(index[:,j:j+1], tmp.shape[-1], axis=1)
    tmp[tmp>=bound] += 1

  return index

def total_dist(index, dist_matrix):
  index = np.hstack((index, index[:,0:1]))
  dist = np.zeros(index.shape[0])
  for i in range(index.shape[-1]-1):
    dist += dist_matrix[index[:,i], index[:,i+1]]

  return dist

def decimal_to_binary(decimal, length):
  binary = np.binary_repr(decimal, length)
  binary = list(map(float, binary))
  
  return np.array(binary)[::-1]

def ackley(x):
  return - 2 * (1 - np.exp(-0.2 * np.abs(x))) - np.e + np.exp(np.cos(10*np.pi*x))

def step(x):
  a = x.copy()
  a[x<4.95] = 0
  a[x>=4.95] = 1

  return a

def beale(x, y):
  out = (1.5 - x + x * y) ** 2 + (2.25 - x + x*y**2) ** 2 + (2.625 - x + x * y**3) ** 2
  return - out

if __name__ == "__main__":
  coding = 0
  generation = 30
  population = 30
  length = 10
  p_cross = 0.8
  p_mute = 0.1
  x_min, x_max = -4, 4
  y_min, y_max = -4, 4
  n_elites = 2

  f = beale

  for type in range(3):
    fit_max = np.array([])
    for i in range(10):
      combi = choice(2, size=(population, length))
      x, y = combi[:,:4], combi[:,4:]

      phenotype_x = ptype(gray_to_binary(x), x_min, x_max)
      phenotype_y = ptype(gray_to_binary(y), y_min, y_max)
      fitness = f(phenotype_x, phenotype_y)

      for j in range(generation):
        elites = elite(combi, fitness, n_elites)
        combi = roulette_wheel(combi, fitness, population-n_elites)
        #x = tournament(x, fitness, population-n_elites, 2)
        combi = crossover(combi, p_cross, TYPE_UNIFORM)
        combi = mutate(combi, p_mute)
        combi = np.vstack((elites, combi))

        x, y = combi[:,:4], combi[:,4:]
        phenotype_x = ptype(gray_to_binary(x), x_min, x_max)
        phenotype_y = ptype(gray_to_binary(y), y_min, y_max)
        fitness = f(phenotype_x, phenotype_y)

      fit_max = np.append(fit_max, fitness.max())
    
    print(fit_max.mean(), fit_max.std())

  cities = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1],
    [1,2],
    [2,1],
    [2,2],
    [2,3],
    [3,2],
    [3,3]
  ])

  #plot_cities(cities)
  dist = cdist(cities, cities)

  for j in range(10):
    order = np.random.randint(100, size=(population, cities.shape[0]))
    for i in range(order.shape[-1]):
      order[:,i] %= order.shape[-1] - i

    simple = simple_repr(order)
    fitness = 1 / total_dist(simple, dist)

    for i in range(generation):
      elites = elite(order, fitness, n_elites)
      order = roulette_wheel(order, fitness, population-n_elites)
      order = crossover(order, p_cross, TYPE_UNIFORM)
      order = mutate(order, p_mute, True)
      order = np.vstack((elites, order))
      simple = simple_repr(order)
      fitness = 1 / total_dist(simple, dist)

    print(1/fitness.max())