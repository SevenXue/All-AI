#! 遗传算法生成环路

import numpy as np
import random
import operator
import pandas as pd

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        x_dis = abs(self.x - city.x)
        y_dis = abs(self.y - city.y)
        distance = np.sqrt((x_dis ** 2) + (y_dis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(0, len(self.route)):
                from_city = self.route[i]
                to_city = None
                if i + 1 < len(self.route):
                    to_city = self.route[i + 1]
                else:
                    to_city = self.route[0]
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness

def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route

def inital_population(pop_size, city_list):
    population = []

    for i in range(0, pop_size):
        population.append(create_route(city_list))
    return population

def rank_routes(population):
    fitness_results = {}

    for i in range(0, len(population)):
        fitness_results[i] = Fitness(population[i]).route_fitness()
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)

def selection(pop_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(pop_ranked), columns=["index", "fitness"])
    df['cum_sum'] = df.fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

    for i in range(0, elite_size):
        selection_results.append(pop_ranked[i][0])

    for i in range(0, len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(0, len(pop_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(pop_ranked[i][0])
                break
    return selection_results

def mating_pool(population, selection_resluts):
    """

    :param population:
    :param selection_resluts:
    :return:
    """
    mating_pool = []
    for i in range(0, len(selection_resluts)):
        index = selection_resluts[i]
        mating_pool.append(population[index])
    return mating_pool

def breed(parent1, parent2):
    child = []
    child_p1 = []
    child_p2 = []

    gene_A = int(random.random() * len(parent1))
    gene_B = int(random.random() * len(parent2))

    start_gene = min(gene_A, gene_B)
    end_gene = max(gene_A, gene_B)

    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])
    child_p2 = [item for item in parent2 if item not in child_p1]

    child = child_p1 + child_p2

    return child

def breed_population(matin_pool, elite_size):
    children = []
    length = len(matin_pool) - elite_size
    pool = random.sample(matin_pool, len(matin_pool))

    for i in range(0, elite_size):
        children.append(matin_pool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matin_pool) - i -1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if(random.random() < mutation_rate):
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1
    return individual

def mutate_population(population, mutation_rate):
    mutate_pop = []

    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutate_pop.append(mutated_ind)

    return mutate_pop

def next_generation(currentgen, elite_size, mutation_rate):
    pop_ranked = rank_routes(currentgen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(currentgen, selection_results)
    children = breed_population(matingpool, elite_size)
    nextgeneration = mutate_population(children, mutation_rate)

    return nextgeneration

def genetic_algorithm(population, pop_size, elite_size, mutation_rate, generations):
    pop = inital_population(pop_size, population)

    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    return best_route

if __name__ == "__main__":
    city_list = []