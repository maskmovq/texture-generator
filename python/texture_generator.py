import numpy as np
from PIL import Image

np.set_printoptions(linewidth=150)

def create_XY(xs, x0, x1, ys, y0, y1):
    Y, X = np.indices((ys,xs), dtype=np.float64)
    X = x0 + (x1-x0)/(xs-1)*X
    Y = y1 + (y0-y1)/(ys-1)*Y
    return X, Y

def normalize(R):
    rmin = np.nanmin(R)
    rmax = np.nanmax(R)
    R = (R-rmin)/(rmax-rmin)
    return R

def save_grayscale(R, path):
    R = (R*255+0.5).astype(np.uint8)
    image = Image.fromarray(R, mode='L')
    image.save(path)

def save_rgb(R, path):
    R = (R*255+0.5).astype(np.uint8)
    image = Image.fromarray(R, mode='RGB')
    image.save(path)
    
#########################################################################

xs = 1000
ys = 500
x0, x1 = -30, 30
y0, y1 = -10, 10

X, Y = create_XY(xs, x0, x1, ys, y0, y1)
R = np.cos(X)*np.sin(Y)

bins = np.array([-100, -0.8, 0, 0.8, 100])
bin_widths = bins[1:] - bins[:-1]
colors0 = np.array([[1,0,0],[0,1,0],[0,0,1],[0.5, 0.5, 0.5]])
colors1 = np.array([[1,0,0],[1,1,0],[0,0,0],[0.5, 0.5, 0.5]])

D = np.digitize(R, bins[1:-1])
C0 = np.take(colors0, D, axis=0)
C1 = np.take(colors1, D, axis=0)
print(R)
print(D)

F = (R - np.take(bins, D))/np.take(bin_widths, D)
F = np.repeat(F, 3)
F.shape = C0.shape
print(F)
C = (1-F)*C0 + F*C1
print(C)
save_rgb(C, "testcolor.png")


#R = normalize(R)
#save_grayscale(R, "test.png")

