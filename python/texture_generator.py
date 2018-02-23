import numpy as np
from PIL import Image

np.set_printoptions(linewidth=150)

xs = 1000
ys = 500
x0, x1 = -30, 30
y0, y1 = -10, 10

Y, X = np.indices((ys,xs), dtype=np.float64)
X = x0 + (x1-x0)/(xs-1)*X
Y = y1 + (y0-y1)/(ys-1)*Y

R = np.cos(X)*np.sin(Y)

rmin = np.nanmin(R)
rmax = np.nanmax(R)
R = (R-rmin)/(rmax-rmin)

# convert to grayscale image

R = (R*255).astype(np.uint8)

image = Image.fromarray(R, mode='L')
image.save("test.png")
