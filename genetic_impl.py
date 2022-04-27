import random
import cv2
import numpy as np
import argparse
import os

# p = 5000, 500, 50
# m = 0.0001, 0.001

population_size = 500
max_iter = 200  # number of generations
mutation_rate = 0.0001
move = [[0, 0], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
cost_values = [0, 45, 90, 135, 180]
best_score, best_result = 0.0, None  # best score and solution of all generations


class Individual:
    """
    chromosome: 1D array for each individual
    fitness_score: float number for each individual's solution
    """
    chromosome = []
    fitness_score = 0

    def __init__(self, dimension, size):
        """
        initialize individuals
        :param dimension: given size of environment exp: 20
        :param size: individual's size exp: 20x20
        condition: matrix of solution with each individual moves
        total_cost: cost of individual moves
        """
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
        """
        calculate how individual is close a given solution via comparison of matrices
        original image and individual's solutions are compared
        :return: individual's fitness score
        """
        score = 0
        for i in range(len(target)):
            if self.condition[i] == target[i]:
                score += 1

        self.fitness_score = score / len(target)

    def crossing_over(self, parent, dimension):
        """
        crossing over for desired individuals
        :param parent: chromosome array to exchange genes
        :param dimension: size of environment to create individual
        :return:
        """
        child = Individual(dimension, size)
        rand = random.randrange(0, len(target))

        for i in range(len(target)):
            if i > rand:
                child.chromosome[i] = self.chromosome[i]
            else:
                child.chromosome[i] = parent.chromosome[i]

        return child

    def mutation(self):
        """
        change each individual's gene with random index and value
        :return:
        """
        for i in range(len(target)):
            if random.random() <= mutation_rate:
                # self.chromosome[i] = random.choice([0, 255])
                self.chromosome[i] = np.random.randint(len(move), size=1)

    @staticmethod
    def save_best(filename, img, p, m, s):
        """
        save the best solution as image file into related folder
        :param filename: given file as original image (target)
        :param img: best solution or individual to given file
        :param p: population size
        :param m: mutation rate
        :param s: score
        :return:
        """
        path = f'results/{filename.split(".")[0]}'
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f'{path}/{p}_{m}_{s}_{filename}', img)

    @staticmethod
    def calculate_cost(next_m, first_m):
        """
        calculate cost of individual
        :param next_m: individual's next step that generated as random
        :param first_m: individual's first step that generated as random
        multiply difference value as index in cost values array
        :return:
        """
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
    parser.add_argument("-f", "--file", default='deneme.jpg', help='image name')
    parser.add_argument("-d", "--dimension", default=20, help='environment size')

    args = vars(parser.parse_args())

    file = args['file']  # get filename
    dim = int(args['dimension'])  # get size of environment

    """
    read image via opencv imread function, convert image to binary format and
    show until generations end
    """
    target = cv2.imread(f"images/{file}", 0).reshape(dim * dim)
    (thresh, target) = cv2.threshold(target, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Original image", target.reshape(dim, dim))

    size = dim * dim  # individual's size
    population = [Individual(dim, size) for i in range(population_size)]  # create population

    iterations = 0
    while True:
        iterations += 1
        # find all individual's fitness score
        for i in population:
            i.fitness()

        local_best = 0  # each generation's bests solution
        for i in population:
            if local_best < i.fitness_score:
                local_best = i.fitness_score
                if best_score == 0.0:
                    best_score = local_best
                # if generated image greater than best, set best score as image
                generated_image = np.array(i.condition.reshape(dim, dim), dtype=np.uint8)
                if best_score < local_best:
                    best_score = local_best
                    best_result = generated_image

        cv2.imshow("Generated image", generated_image)  # show solution
        cv2.waitKey(1)

        if iterations == max_iter:
            cv2.waitKey(1)
            break

        print("Generation:", iterations + 1, "\tScore:", str(round(local_best, 4)))
        print("Current best score:", best_score)
        new_population = []

        """
        the higher the fitness function of an individual, the more it takes place in the new population
        """
        for i in range(len(population)):
            n = int(population[i].fitness_score * 100) + 1

            for j in range(n):
                new_population.append(population[i])

        """
        crossing over and mutations are done here
        """
        for i in range(len(population)):
            i1 = random.randrange(0, len(new_population))
            i2 = random.randrange(0, len(new_population))

            new_i1 = new_population[i1]
            new_i2 = new_population[i2]

            child = new_i1.crossing_over(new_i2, dim)
            child.mutation()

            population[i] = child

    # save global best image (max score) for all generations
    Individual.save_best(filename=file, img=best_result, p=population_size, m=mutation_rate, s=best_score)
