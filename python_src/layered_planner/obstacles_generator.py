import numpy as np

if __name__ == '__main__':
    width = 0.2
    length = 0.8
    loops = 20
    for i in range(loops):
        x = np.random.uniform(-1.7, 2.5)
        y = np.random.uniform(-2.5, 1.7)
        print('np.array([[%lf, %lf],[%lf, %lf],[%lf, %lf],[%lf, %lf]]),' % (x - length, y, x, y, x, y + width, x - length, y + width))
        print('np.array([[%lf, %lf],[%lf, %lf],[%lf, %lf],[%lf, %lf]]),' % (x - width, y, x, y, x, y + length, x - width, y + length))
