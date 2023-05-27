import numpy as np

if __name__ == '__main__':
    t = 1000
    alpha = 0.9
    T_min = 0.1
    for i in range(100):
        t = t * alpha
        print(t)
    # 画出这个函数
    import matplotlib.pyplot as plt
    x = np.linspace(0, 100, 1000)
    y = 1000 * 0.9 ** x
    plt.plot(x, y)
    # plt.xlim(0, 100)
    # plt.ylim(0, 1000)
    # 画出label
    plt.xlabel('iteration')
    plt.ylabel('temperature')
    plt.title('Simulated Annealing')
    plt.grid(True)
    plt.show()


    # # 计算a相比b有多大的提升
    # a = 280.45
    # b = 291.09
    # print((b - a) / a)

    # width = 0.2
    # length = 0.8
    # loops = 20
    # for i in range(loops):
    #     x = np.random.uniform(-1.7, 2.5)
    #     y = np.random.uniform(-2.5, 1.7)
    #     print('np.array([[%lf, %lf],[%lf, %lf],[%lf, %lf],[%lf, %lf]]),' % (x - length, y, x, y, x, y + width, x - length, y + width))
    #     print('np.array([[%lf, %lf],[%lf, %lf],[%lf, %lf],[%lf, %lf]]),' % (x - width, y, x, y, x, y + length, x - width, y + length))
