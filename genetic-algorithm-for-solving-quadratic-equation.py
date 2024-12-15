import random

# Quadratic equation: f(x) = x^2 - 3x + 2
a = 1
b = -3
c = 2


# Fitness function (evaluates how close the given value is to the real solution of the equation)
# This should approach 0
def fitness(x):
    return abs(a * x**2 + b * x + c)


# Initial population generation
def generate_population(size, lower_bound, upper_bound):
    return [random.uniform(lower_bound, upper_bound) for _ in range(size)]


# Selection (Selecting individuals with low fitness)
# Combines the individuals in the population with their fitness values,
# then sorts them in ascending order based on their fitness values
# and returns the first number_of_parents individuals
def select(population, fitness_values, number_of_parents):
    parents = sorted(zip(population, fitness_values), key=lambda x: x[1])[:number_of_parents]
    return [p[0] for p in parents]


# Crossover: Creating two new individuals from two parents
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            # Arithmetic crossover method (averaging)
            parent1 = parents[i]
            parent2 = parents[i + 1]
            offspring1 = (parent1 + parent2) / 2
            offspring2 = (parent1 - parent2) / 2
            offspring.extend([offspring1, offspring2])
    return offspring


# Mutation: Random mutation of an individual
def mutate(offspring, mutation_chance, lower_bound, upper_bound):
    for i in range(len(offspring)):
        if random.random() < mutation_chance:
            offspring[i] += random.uniform(-1, 1)
            # Ensure the mutation keeps the individual within the bounds
            offspring[i] = max(min(offspring[i], upper_bound), lower_bound)
    return offspring


def genetic_algorithm(population_size=100, generations=50, mutation_chance=0.05, lower_bound=-10, upper_bound=10):
    # Step 1: Generate initial population
    population = generate_population(population_size, lower_bound, upper_bound)

    for generation in range(generations):
        # Step 2: Evaluate fitness values
        fitness_values = [fitness(x) for x in population]

        # Step 3: Selection (select the best 50% of individuals)
        number_of_parents = population_size // 2
        parents = select(population, fitness_values, number_of_parents)

        # Step 4: Crossover (create offspring from the parents)
        offspring = crossover(parents)

        # Step 5: Introduce mutations
        offspring = mutate(offspring, mutation_chance, lower_bound, upper_bound)

        # Step 6: Create a new population (parents + their offspring)
        population = parents + offspring

        # Check if any solution is good enough (fitness value close to 0)
        best_fitness = min(fitness_values)
        best_solution = population[fitness_values.index(best_fitness)]
        print(f"Generation {generation}: Best solution x = {best_solution}, fitness = {best_fitness}")

        # Stop if the solution is very close to 0
        if best_fitness < 1e-6:
            break

    best_fitness = min(fitness_values)
    best_solution = population[fitness_values.index(best_fitness)]
    return best_solution


solution = genetic_algorithm()
print(f"x = {solution} solution found")
