import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from LOAD_SAVE import *
X_train=load("X_train")
X_val=load("X_test")
y_train=load("y_train")
y_val =load("X_test")

#import numpy as np

def fitness_function(solution):

    return np.sum(np.square(solution))

def hswph_optimizer(fitness_func, dim, bounds, n=20, max_iter=50):
    LB, UB = bounds
    population = np.random.uniform(LB, UB, size=(n, dim))
    fitness = np.array([fitness_func(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n):
            xi = population[i]

            ## Phase 1: Exploration (Hunting/Nesting or Escape Strategy)
            if np.random.rand() < 0.5:
                j, k = np.random.choice(n, 2, replace=False)
                xj, xk = population[j], population[k]
                adaptive = np.random.rand()
                V = adaptive * (xj - xk)  # forward velocity
                xi = xi + V * np.random.randn(dim)  # hunting update
            else:
                escape = np.random.uniform(-1, 1, size=dim)
                xi = xi + escape + np.random.randn(dim) * (best_solution - xi)

            ## Phase 2: Exploitation (Following/Nesting)
            if np.random.rand() < 0.5:
                xi = xi + np.random.rand() * (best_solution - xi)
            else:
                xi = xi + np.random.randn(dim) * (xi - best_solution)

            ## Phase 3: Nesting
            if np.random.rand() < 0.5:
                xi += np.random.normal(0, 1, dim)  # random nesting step
            else:
                # Levy flight approximation
                beta = 1.5
                sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                         (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
                levy = np.random.randn(dim) * sigma
                xi = xi + levy * (xi - best_solution)

            ## Phase 4: Mating Behavior
            male, female = population[np.random.choice(n, 2, replace=False)]
            crossover = 0.5 * (male + female) + np.random.rand(dim) * (male - female)
            xi = crossover

            ## Phase 5: Hunter-Prey Interaction
            mean_pos = np.mean(population, axis=0)
            if np.random.rand() < 0.5:  # Prey
                xi = xi + np.random.rand() * (xi - mean_pos)
            else:  # Hunter
                xi = xi + np.random.rand() * (mean_pos - xi)

            ## Bound and update
            xi = np.clip(xi, LB, UB)
            f_xi = fitness_func(xi)

            if f_xi < fitness[i]:
                population[i] = xi
                fitness[i] = f_xi

                if f_xi < best_fitness:
                    best_solution = xi
                    best_fitness = f_xi

        ## Phase 6: Population Reduction
        sort_idx = np.argsort(fitness)
        population = population[sort_idx]
        fitness = fitness[sort_idx]

        # Elitism + Remove worst
        retain = int(n * 0.8)
        offspring = np.random.uniform(LB, UB, size=(n - retain, dim))
        population = np.vstack((population[:retain], offspring))
        fitness = np.array([fitness_func(ind) for ind in population])
        n = population.shape[0]

        print(f"Iter {t+1}/{max_iter}, Best Fitness: {best_fitness:.5f}")

    return best_solution, best_fitness
dim = 10
bounds = (-5.0, 5.0)
best_sol, best_fit = hswph_optimizer(fitness_function, dim, bounds, n=30, max_iter=100)
print("Best Solution:", best_sol)
print("Best Fitness:", best_fit)