import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy.random as rd
import os

move = [[0, 0], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]]
cost_values = [0, 45, 90, 135, 180]
# 5k iter 500de bir


class Population:
    def __init__(self, original_img, mutation_rate, population_size, individual_len, iteration, n):
        self.original_img = original_img
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.individual_len = individual_len
        self.iteration = iteration
        self.start = [n - 1, 0]
        self.population = np.random.randint(len(move), size=(population_size, individual_len))

    def draw(self, condition, individual, dim):
        """
        :param dim:
        :param condition: matrisin durumu
        :param individual: birey adimlari
        :return:
        """
        start = self.start
        condition[start[0], start[1]] = 0

        # i = bireyin adimlari
        # total_cost = 0
        prev = [0, 0]
        for i in individual:
            act = [x + y for x, y in zip(move[i], start)]
            if not (act[0] < 0 or act[1] < 0 or act[0] > dim - 1 or act[1] > dim - 1):
                # total_cost += self.calculate_cost(move[i], prev)
                start = act
                prev = move[i]
                condition[act[0], act[1]] = 0

        return condition

    # @staticmethod
    # def calculate_cost(next_m, first_m):
    #     diff = []
    #     zip_object = zip(next_m, first_m)
    #     for i, j in zip_object:
    #         diff.append(abs(i - j))
    #
    #     if (diff[0] == 0 and diff[1] != 0) or (diff[0] != 0 and diff[1] == 0):
    #         if first_m == [0, 0]:
    #             return cost_values[0]
    #         return cost_values[4]
    #
    #     return cost_values[sum(diff)]

    def fitness_function(self, condition, dim):
        """
        fitness function for each individual performance
        :param condition: individual's last condition
        :param dim:
        :return:
        """
        similarity = 0
        for i in range(dim):
            for j in range(dim):
                if condition[i][j] == self.original_img[i][j]:
                    similarity += 1
        # similarity = np.sum(self.original_img == condition)
        return round(similarity / (dim * dim), 4)
        # cost = round(cost / (180 * (self.individual_len - 1)), 4)

    def crossing_over(self, i1, i2):
        temp = [random.randint(0, 1) for _ in range(self.individual_len)]
        child1 = i1
        child2 = i2
        for i in range(self.individual_len):
            if temp[i] == 0:
                tmp = child1[i]
                child1[i] = child2[i]
                child2[i] = tmp

        return child1, child2

    def sort_population(self, weights):
        population_dict = {}
        for i in range(self.population_size):
            population_dict[i] = weights[i]
        sorted_dict = {k: v for k, v in sorted(population_dict.items(), key=lambda item: item[1], reverse=True)}

        return sorted_dict

    def selection(self, fitness):
        # n = int(self.population_size / 4)  # sample size
        max_f = sum(fitness)
        selection_probs = fitness / max_f
        return random.choices(list(range(0, self.population_size)), weights=selection_probs, k=self.population_size)
        # return self.population[rd.choice(self.population_size, )]
        # selected = []
        # size = self.population_size * (self.population_size + 1) / 2
        # for i in range(self.population_size):
        #     for j in range(self.population_size - i):
        #         selected.append(sorted_indexes[i])
        #
        # random.shuffle(selected)
        # random_index = random.randint(0, size - 1)
        #
        # random_select = [selected[random_index]]
        # for i in range(self.population_size - 1):
        #     random_index = random.randint(0, size - 1)
        #     while random_select[-1] == selected[random_index]:
        #         random_index = random.randint(0, size - 1)
        #     random_select.append(selected[random_index])
        #
        # return random_select

    # @staticmethod
    # def find_best(sorted_dict):
    #     return list(sorted_dict.keys())[0], list(sorted_dict.values())[0]

    def mutation(self, i):
        for j in range(int(self.individual_len * self.mutation_rate)):
            index = random.randint(0, self.individual_len - 1)
            i[index] = random.randint(0, len(move) - 1)  # 0-8 arasi sayi

    @staticmethod
    def save_best(condition, dim, filename, k):
        # i = self.population[index]
        # condition = np.ones([dim, dim], dtype=int)

        # cond = self.draw(condition, i, dim)
        # print(cond)
        path = f'results/{filename.split(".")[0]}'
        os.makedirs(path, exist_ok=True)
        # plt.imsave(f'{path}/{"50lik" + filename}', np.array(condition).reshape(dim, dim), cmap=cm.gray)
        cv2.imwrite(f'{path}/{filename}', condition)

    # @staticmethod
    # def get_grid():
    #     plt.figure(1)
    #     plt.grid(True)
    #     return plt


