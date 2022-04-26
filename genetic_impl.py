import random
import cv2
import numpy as np
import argparse
import os

population_size = 5000
max_iter = 400
mutation_rate = 0.0001
move = [[0, 0], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
cost_values = [0, 45, 90, 135, 180]


class Individual:
    chromosome = []
    fitness_score = 0

    def __init__(self, dimension, size):
        self.chromosome = np.random.randint(len(move), size=size)
        self.condition = np.full((dimension, dimension), 255, dtype=int)
        self.total_cost = 0

        prev = [0, 0]
        start = [dimension - 1, 0]
        self.condition[start[0], start[1]] = 0

        for i in self.chromosome:
            act = [x + y for x, y in zip(move[i], start)]
            if not (act[0] < 0 or act[1] < 0 or act[0] > dim - 1 or act[1] > dim - 1):
                self.total_cost += self.calculate_cost(move[i], prev)
                start = act
                prev = move[i]
                self.condition[act[0], act[1]] = 0
        self.condition = self.condition.reshape(dim * dim)

    def fitness(self):
        score = 0
        for i in range(len(target)):
            if self.condition[i] == target[i]:
                score += 1

        self.fitness_score = score / len(target)

    def crossing_over(self, obj, dimension):
        child = Individual(dimension, size)
        breakpoint = random.randrange(0, len(target))

        for i in range(len(target)):
            if i > breakpoint:
                child.chromosome[i] = self.chromosome[i]
            else:
                child.chromosome[i] = obj.chromosome[i]

        return child

    def mutation(self):
        for i in range(len(target)):
            if random.random() <= mutation_rate:
                self.chromosome[i] = random.choice([0, 255])

    @staticmethod
    def save_best(filename, img, p, m):
        path = f'results/{filename.split(".")[0]}'
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f'{path}/{p}_{m}_{filename}', img)

    @staticmethod
    def calculate_cost(next_m, first_m):
        diff = []
        zip_object = zip(next_m, first_m)
        for i, j in zip_object:
            diff.append(abs(i - j))

        if (diff[0] == 0 and diff[1] != 0) or (diff[0] != 0 and diff[1] == 0):
            if first_m == [0, 0]:
                return cost_values[0]
            return cost_values[4]

        return cost_values[sum(diff)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default='circle.png', help='image name')
    parser.add_argument("-d", "--dimension", default=20, help='environment size')

    args = vars(parser.parse_args())

    file = args['file']
    dim = int(args['dimension'])

    target = cv2.imread(f"images/{file}", 0).reshape(dim * dim)
    (thresh, target) = cv2.threshold(target, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Original image", target.reshape(dim, dim))

    size = dim * dim
    population = [Individual(dim, size) for i in range(population_size)]

    iterations = 0
    while True:
        iterations += 1
        for i in population:
            i.fitness()

        best = 0
        for i in population:
            if best < i.fitness_score:
                best = i.fitness_score
                generated_image = np.array(i.condition.reshape(dim, dim), dtype=np.uint8)
                i.save_best(filename=file, img=generated_image, p=population_size, m=mutation_rate)

        cv2.imshow("Generated image", generated_image)
        cv2.waitKey(1)

        if iterations == max_iter:
            cv2.waitKey(1)
            break

        print("Generation:", iterations + 1, "\tScore:", str(round(best, 4)))
        new_population = []

        for i in range(len(population)):
            n = int(population[i].fitness_score * 100) + 1

            for j in range(n):
                new_population.append(population[i])

        for i in range(len(population)):
            i1 = random.randrange(0, len(new_population))
            i2 = random.randrange(0, len(new_population))

            new_i1 = new_population[i1]
            new_i2 = new_population[i2]

            child = new_i1.crossing_over(new_i2, dim)
            child.mutation()

            population[i] = child
