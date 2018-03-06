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

bins = np.array([-1,-0.9, 0.1, 0.9, 1])
color_values = np.array([
[[0.3, 0.4, 0.5],[1.0, 1.0, 0],[0,  0.6, 0.4],[0.8, 0.1, 0]],
[[0.3, 0.4, 0.5],[1.0, 0, 1.0],[0.5,0.1, 0.4],[0.8, 0.1, 0]]
])

D = np.digitize(R, bins[1:-1])
bin_widths = bins[1:] - bins[:-1]
print(bin_widths)
print(D)
C0 = np.take(color_values[0], D, axis = 0)
C1 = np.take(color_values[1], D, axis = 0)

F = (R - np.take(bins, D)) / np.take(bin_widths, D)
print(F.shape)
print(F)
#print(np.broadcast_to(F, (5,10,3)))
F = np.repeat(F, 3)
F.shape = C0.shape

#print(F.T*C1.T)
C = (1-F)*C0 + F*C1
'''
print(R2)
print(R2.shape)
R3 = np.empty((3, R2.shape[0], R2.shape[1]), dtype = np.float32)
R3[0,:,:]=R2[:,:,0]
R3[1,:,:]=R2[:,:,1]
R3[2,:,:]=R2[:,:,2]
print(R3)
'''


Image.fromarray((C0*255+0.5).astype(np.uint8), mode='RGB').save("testc0.png")
Image.fromarray((C1*255+0.5).astype(np.uint8), mode='RGB').save("testc1.png")
Image.fromarray((C*255+0.5).astype(np.uint8), mode='RGB').save("testc.png")


R = (R-rmin)/(rmax-rmin)

# convert to grayscale image

R = (R*255).astype(np.uint8)

image = Image.fromarray(R, mode='L')
image.save("test.png")
