from PIL import Image
import numpy as np
from population import Population
import argparse
import cv2

MUTATION = 0.0001
POPULATION = 5000
ITERATION = 100


def binary_image(filename, resize):
    img = Image.open(f'images/{filename}').convert('L')
    img = img.resize((resize, resize))
    img.save(f'images/{filename}')

    np_img = np.array(img)
    np_img = ~np_img  # invert B&W

    np_img[np_img > 0] = 1  # original image
    return np_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", default='circle.png', help='image name')
    parser.add_argument("-d", "--dimension", default=5, help='environment size')

    args = vars(parser.parse_args())

    file = args['file']
    dim = int(args['dimension'])

    individual_len = dim * (dim - 1)
    original_img = binary_image(file, dim)
    # original_img = [[0, 0, 0, 0, 0],
    #                 [0, 1, 1, 1, 0],
    #                 [0, 1, 0, 1, 0],
    #                 [0, 1, 1, 1, 0],
    #                 [0, 0, 0, 0, 0]
    #                 ]

    p = Population(original_img, MUTATION, POPULATION, individual_len, ITERATION, dim)

    b1 = np.zeros((ITERATION,), dtype=float)  # best ind
    f1 = np.zeros((POPULATION,), dtype=float)  # fitness
    # plt = p.get_grid()
    # c = int(POPULATION - (POPULATION / 2))  # copy rate

    target = cv2.imread(f"images/{file}", cv2.IMREAD_GRAYSCALE).reshape(dim * dim)
    (thresh, im_bw) = cv2.threshold(target, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("Original image", im_bw.reshape(dim, dim))

    for generation in range(ITERATION):
        k = 0
        best_fitness, best_condition = 0.0, None

        for i in p.population:
            condition = np.full((dim, dim), 255, dtype=int)
            # condition_i, cost_i = p.draw(condition, i, dim)  # bir bireyin gezme matrisi
            condition_i = p.draw(condition, i, dim)  # bir bireyin gezme matrisi

            fitness_ind = p.fitness_function(condition_i, dim)
            f1[k] = fitness_ind

            if fitness_ind > best_fitness:
                best_condition = condition_i
                best_fitness = fitness_ind

            # plt.imshow(condition, interpolation='nearest', cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.00001)
            # plt.close()

            k += 1

        sorted_population = p.sort_population(f1)
        # best_index, best_value = p.find_best(sorted_population)

        b1[generation] = best_fitness

        # generated_img = np.array(best_condition.reshape(dim, dim), dtype=np.uint8)
        # cv2.imshow("Generated image", generated_img)
        # cv2.waitKey(1)

        # found = True
        # for i in range(dim * dim):
        #     if not generated_img.reshape(dim * dim)[i] == target[i]:
        #         found = False

        # if found:
        #     cv2.waitKey(0)
        #     break

        # exit(4)
        selected_indexes = p.selection(f1)

        new_population = []
        for i in range(0, p.population_size, 2):
            new_i1, new_i2 = p.crossing_over(p.population[selected_indexes[i]], p.population[selected_indexes[i + 1]])
            new_population.append(new_i1)
            new_population.append(new_i2)

        # k = 0
        # for i in range(c, p.population_size):
        #     copied_i = p.population[selected_indexes[k]]
        #     new_population[i] = copied_i
        #     k += 1

        p.population = new_population
        for i in p.population:
            p.mutation(i)

        generated_img = np.array(best_condition.reshape(dim, dim), dtype=np.uint8)
        cv2.imshow("Generated image", generated_img)
        cv2.waitKey(1)

        print('Generation:', generation, "\t", 'Score:', b1[generation])
        if generation == ITERATION - 1:
            p.save_best(best_condition, dim, file, generation)


