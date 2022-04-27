from PIL import Image
import numpy as np
img = Image.open(f'images/circle.png').convert('L')
image = np.array(img)
for i in image:
    print(i)
exit(3)
img = img.resize((10, 10))
