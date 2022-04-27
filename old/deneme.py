import random
from time import sleep
import cv2
import numpy as np
from PIL import Image

size1 = 10
size2 = 10


class Individual:
    genes = []
    fitness_score = 0

    def __init__(self):
        self.genes = np.array(
            [random.choice([0, 255]) for i in range(size1 * size2)]
        )

    def fitness(self):
        score = 0
        for i in range(len(target)):
            print(self.genes[i], target[i])
            if self.genes[i] == target[i]:
                score += 1

        self.fitness_score = score / len(target)

    def crossing_over(self, obj):
        child = Individual()
        breakpoint = random.randrange(0, len(target))

        for i in range(len(target)):
            if i > breakpoint:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = obj.genes[i]

        return child

    def mutation(self):
        for i in range(len(target)):
            if random.random() <= mutation_rate:
                self.genes[i] = random.choice([0, 255])

    def get_phrase(self):
        return "".join(self.genes)


population_size = int(input("Population: "))

# # load image
# loaded_image = cv2.imread("images/circle.png", 0)
# # resize image
# colored_img = cv2.resize(loaded_image, (size1, size2), cv2.INTER_NEAREST)


loaded_image = Image.open(f'images/circle.png').convert('L')
resized_img = loaded_image.resize((size1, size2))

# convert binary
target = np.array(resized_img)
target = ~target  # invert B&W
target[target > 0] = 1  # original image

print(target.shape)
print(target)
# target = cv2.imread("images/circle.png", 0).reshape(size1 * size2)
# target = cv2.imread("images/circle.png", 0).resize(11 * 12)

cv2.imshow("Original image", target.reshape(size1, size2))

population = [Individual() for i in range(population_size)]
mutation_rate = 0.0001

iterations = 0
while True:
    iterations += 1
    for i in population:
        i.fitness()

    best = 0
    for i in population:
        if best < i.fitness_score:
            best = i.fitness_score
            phrase = np.array(i.genes.reshape(size1, size2), dtype=np.uint8)
    print(
        "Generation:",
        iterations,
        "  Score:",
        str(best * 100)[:5],
    )
    cv2.imshow("Generated image", phrase)
    cv2.waitKey(1)

    found = True
    for i in range(11 * 12):
        if not phrase.reshape(size1 * size2)[i] == target[i]:
            found = False

    if found:
        cv2.waitKey(0)
        break

    matingPool = []

    for i in range(len(population)):
        n = int(population[i].fitness_score * 100) + 1

        for j in range(n):
            matingPool.append(population[i])

    for i in range(len(population)):
        a = random.randrange(0, len(matingPool))
        b = random.randrange(0, len(matingPool))

        parent_a = matingPool[a]
        parent_b = matingPool[b]

        child = parent_a.crossing_over(parent_b)
        child.mutation()

        population[i] = child

input()
