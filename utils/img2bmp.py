from PIL import Image
import numpy as np

img = Image.open('image_barcelona_512.jpg')
ary = np.array(img)

im = Image.fromarray(ary.astype(np.uint8))
im.save('road.bmp')
