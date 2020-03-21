import numpy as np
from matplotlib import pyplot as plt
from plot_utilities import matplotlib_nikita_style
from operator import add
from matplotlib.ticker import MaxNLocator

matplotlib_nikita_style()

def plot_grid(grid_m=7, grid_n=5):
    # creating a grid
    x_values = np.array(range(0, grid_m))
    y_values = np.array(range(0, grid_n))
    xx, yy = np.meshgrid(x_values, y_values)
    plt.plot(xx, yy, marker='s', markersize='25', color='k', linestyle='none')
    plt.xlim([-1, grid_m])
    plt.ylim([-1, grid_n])
    plt.xlabel('m')
    plt.ylabel('n')
    # plot start and end
    start_end = np.array([[0.5, 0.5], [5.5, 3.5]])
    plt.plot(start_end[:10, 0], start_end[:10, 1], "s", color='red', markersize=5)
    plt.savefig('figures/grid.pdf')


def draw_line(paths, color='dodgerblue'):
    xx = list(paths[::2])
    yy = list(paths[1:][::2])

    for i in range(len(xx)-1):
            plt.plot([xx[i], xx[i+1]], [yy[i], yy[i+1]], '-', linewidth=2, color=color)


def exp_dist():
    lam = 5
    beta = 1/lam
    list = np.random.exponential(scale=beta, size=10)
    return list


def put_points(paths):
    xx = list(paths[::2])
    yy = list(paths[1:][::2])
    points = []
    for i in range(len(xx)-1):
        if np.random.exponential(scale=1 / 5, size=1) > 0.2:
            point = np.random.rand(1, 1)
        else:
            point = 0

        if xx[i] == xx[i+1] and point != 0:
            points.append(xx[i])
            points.append(yy[i] + float(point))
        elif yy[i] == yy[i + 1] and point != 0:
            points.append(xx[i] + float(point))
            points.append(yy[i])
    return points


def check_links(points):
    points = [0.5, 0.5] + points + [5.5, 3.5]
    xx = list(points[::2])
    yy = list(points[1:][::2])
    links = []
    for i in range(len(xx) - 1):
        manhattan_dist = abs(xx[i+1] - xx[i]) + abs(yy[i + 1] - yy[i])
        if manhattan_dist > 3:
            links.append(0)
        elif yy[i] != yy[i + 1] and xx[i] != xx[i + 1] and manhattan_dist > 1:
            links.append(0)
        else:
            links.append(1)
    return links



if __name__ == "__main__":
    grid_m = 7
    grid_n = 5
    min_number_of_moves = (grid_m - 1) * (grid_n - 1)
    border_m = grid_m - 1.5
    border_n = grid_n - 1.5

    plot_grid(grid_m, grid_n)

    paths = np.empty((min_number_of_moves, 0))
    for i in range(3):
        point = [0.5, 0.5]
        paths = np.append(paths, point)
        flag = 0
        while flag != 1:
            coin = np.random.randint(2)
            if point[0] < border_m and coin == 0:
                point = list(map(add, point, [1, 0]))
                paths = np.append(paths, point)
            if point[1] < border_n and coin == 1:
                point = list(map(add, point, [0, 1]))
                paths = np.append(paths, point)
            if point[0] == border_m and point[1] == border_n:
                flag = 1

    paths = paths.reshape(3, min_number_of_moves - 6)
    draw_line(paths[0], color='dodgerblue')
    #draw_line(paths[1], color='red')
    #draw_line(paths[2], color='green')

    #exp_list = exp_dist()

    # plot drones
    points = put_points(paths[0])
    xx = list(points[::2])
    yy = list(points[1:][::2])
    plt.plot(xx, yy, "*", color='dodgerblue', markersize=10)
    plt.savefig('figures/grid_w_drones.pdf')

    # plot links
    links = check_links(points)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(False)
    plt.bar(np.arange(len(links)), links, align='center')
    plt.ylim([0, 1.5])
    plt.xlabel('Links between nodes')
    plt.ylabel('Connection/No connection')
    plt.savefig('figures/links.pdf')


plt.show()
