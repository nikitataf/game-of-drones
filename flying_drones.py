import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import norm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
from plot_utilities import matplotlib_nikita_style

matplotlib_nikita_style()


class WallIntersection:
    def __init__(self):
        self.c_x = 4.7
        self.c_y = 0.7
        self.d_x = 5.3
        self.d_y = 0.7

    def cclockwise(self, a_x, a_y, b_x, b_y, c_x, c_y):
        """
        The function determines if three points are listed in a counterclockwise order.
        :param a_x: Segment 1, point A
        :param a_y: Segment 1, point A
        :param b_x: Segment 1, point B
        :param b_y: Segment 1, point B
        :param c_x: Segment 2, point C
        :param c_y: Segment 2, point C
        :return: True or False
        """
        return (c_y - a_y) * (b_x - a_x) > (b_y - a_y) * (c_x - a_x)

    def intersect(self, a_x, a_y, b_x, b_y, c_x, c_y, d_x, d_y):
        """
        Return true if line segments AB and CD intersect
        :param a_x: Segment 1, point A
        :param a_y: Segment 1, point A
        :param b_x: Segment 1, point B
        :param b_y: Segment 1, point B
        :param c_x: Segment 2, point C
        :param c_y: Segment 2, point C
        :param d_x: Segment 2, point D
        :param d_y: Segment 2, point D
        :return: True or False
        """
        return self.cclockwise(a_x, a_y, c_x, c_y, d_x, d_y) != self.cclockwise(b_x, b_y, c_x, c_y, d_x, d_y) \
               and self.cclockwise(a_x, a_y, b_x, b_y, c_x, c_y) != self.cclockwise(a_x, a_y, b_x, b_y, d_x, d_y)


def draw_line(paths, color='dodgerblue'):
    for i in range(len(paths)-1):
        plt.plot([paths[i][0], paths[i+1][0]], [paths[i][1], paths[i+1][1]], '-', linewidth=2, color=color)


def check_links(frames_array):
    links = []
    wall = WallIntersection()
    for timestep in range(len(frames_array)):
        time_stamp = [5.5, 3.5] + list(frames_array[timestep]) + [0.5, 0.5]
        xx = list(time_stamp[::2])
        yy = list(time_stamp[1:][::2])
        buff = []
        for i in range(len(xx) - 1):
            manhattan_dist = abs(xx[i+1] - xx[i]) + abs(yy[i + 1] - yy[i])
            dist = norm(np.array([xx[i+1] - xx[i], yy[i+1] - yy[i]]))
            if dist >= 2:
                buff.append(0)
            #elif yy[i] != yy[i + 1] and xx[i] != xx[i + 1] and dist > 0.72:
            #    buff.append(0)
            elif building_heights[5]>50 and \
                    wall.intersect(xx[i], yy[i], xx[i+1], yy[i+1], c_x=4.7, c_y=0.7, d_x=5.3, d_y=0.7):
                buff.append(0)
            else:
                buff.append(1)
        if 0 in buff:
            links.append(0)
        else:
            links.append(1)
    return links


def init():
    return drone_1


def data_gen():
    mytuple = tuple(range(len(frames_1)))
    counts = iter(mytuple)
    return counts


def run(frame):
    x_1, y_1 = frames_1[frame]
    drone_1.set_data(x_1, y_1)
    x_2, y_2 = frames_2[frame]
    drone_2.set_data(x_2, y_2)
    x_3, y_3 = frames_3[frame]
    drone_3.set_data(x_3, y_3)
    x_4, y_4 = frames_4[frame]
    drone_4.set_data(x_4, y_4)
    x_5, y_5 = frames_5[frame]
    drone_5.set_data(x_5, y_5)
    x_6, y_6 = frames_6[frame]
    drone_6.set_data(x_6, y_6)
    x_7, y_7 = frames_7[frame]
    drone_7.set_data(x_7, y_7)
    x_8, y_8 = frames_8[frame]
    drone_8.set_data(x_8, y_8)
    x_9, y_9 = frames_9[frame]
    drone_9.set_data(x_9, y_9)
    return drone_1


if __name__ == "__main__":
    # drones path
    drone_coords = [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5], [4.5, 0.5], [5.5, 0.5],
                    [5.5, 1.5], [5.5, 2.5], [5.5, 3.5]]

    # creating a grid
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_line(drone_coords)
    drone_1, = ax.plot([0], [0], 'ro')
    drone_2, = ax.plot([0], [0], 'ro')
    drone_3, = ax.plot([0], [0], 'ro')
    drone_4, = ax.plot([0], [0], 'ro')
    drone_5, = ax.plot([0], [0], 'ro')
    drone_6, = ax.plot([0], [0], 'ro')
    drone_7, = ax.plot([0], [0], 'ro')
    drone_8, = ax.plot([0], [0], 'ro')
    drone_9, = ax.plot([0], [0], 'ro')
    x_values = np.array(range(0, 7))
    y_values = np.array(range(0, 5))
    xx, yy = np.meshgrid(x_values, y_values)
    plt.plot(xx, yy, marker='s', markersize='25', color='k', linestyle='none')
    plt.plot([0.5, 5.5], [0.5, 3.5], 'k*', markersize='10')
    plt.xlim([-1, 7])
    plt.ylim([-1, 5])
    plt.xlabel('m')
    plt.ylabel('n')
    ax.grid()
    a, m = 3., 2.  # shape and mode
    np.random.seed(0)
    building_heights = (np.random.pareto(a, 35) + 1) * m
    building_heights = building_heights * 500 / np.linalg.norm(building_heights)

    # making frames
    zero_coord = []
    for i in range(1000):
        zero_coord.append([0.5, 0.5])
    end_coord = []
    for i in range(1000):
        end_coord.append([5.5, 3.5])

    steps = np.arange(0, 1, 0.01)
    frames = []
    for i in range(len(drone_coords) - 1):
        x, y = drone_coords[i]
        x_next, y_next = drone_coords[i + 1]
        if x!=x_next:
            for j in steps:
                frames.append([x + j, y])
        else:
            for j in steps:
                frames.append([x, y + j])
    frames_1 = frames + end_coord[0:900]
    frames_2 = zero_coord[0:100] + frames + end_coord[0:800]
    frames_3 = zero_coord[0:250] + frames + end_coord[0:650]
    frames_4 = zero_coord[0:300] + frames + end_coord[0:600]
    frames_5 = zero_coord[0:475] + frames + end_coord[0:425]
    frames_6 = zero_coord[0:500] + frames + end_coord[0:400]
    frames_7 = zero_coord[0:600] + frames + end_coord[0:300]
    frames_8 = zero_coord[0:700] + frames + end_coord[0:200]
    frames_9 = zero_coord[0:850] + frames + end_coord[0:50]

    # checking links
    frames_1 = np.array(frames_1)
    frames_2 = np.array(frames_2)
    frames_3 = np.array(frames_3)
    frames_4 = np.array(frames_4)
    frames_5 = np.array(frames_5)
    frames_6 = np.array(frames_6)
    frames_7 = np.array(frames_7)
    frames_8 = np.array(frames_8)
    frames_9 = np.array(frames_9)
    frames_array = np.concatenate((frames_1, frames_2, frames_3, frames_4, frames_5,
                                   frames_6, frames_7, frames_8, frames_9), axis=1)

    ani = animation.FuncAnimation(fig, run, data_gen, init_func=init, interval=10)

    # check connectivity
    links = check_links(frames_array)
    # plot connectivity link
    figure, axes = plt.subplots(figsize=(6, 4))
    # figure.subplots_adjust(left=.1, bottom=.1, right=0.97, top=.95)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(True)
    axes.spines['left'].set_visible(True)
    figure.add_axes(axes)
    plt.plot(links, color='#006BB2', linewidth=3)
    #plt.legend(loc='best', facecolor=(0.96, 0.98, 0.99))
    plt.xlabel('Time, [time_steps]')
    plt.ylabel('Connectivity, [1/0]')
    plt.ylim([-0.5, 1.5])
    plt.savefig('figures/connectivity.pdf')

    fig, ax = plt.subplots()
    x = np.arange(0, 1700, 1)
    line, = plt.plot(links, color='#006BB2', linewidth=3)
    plt.xlabel('Time, [time_steps]')
    plt.ylabel('Connectivity, [1/0]')

    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        return line,

    def animate(i):
        line.set_data(x[:i], links[:i])  # update the data.
        return line,

    anim= animation.FuncAnimation(fig, animate, init_func=init, interval=2)

    # 3D plot of city
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    x3 = np.concatenate((xx[0], xx[1], xx[2], xx[3], xx[4]), axis=0)
    y3 = np.concatenate((yy[0], yy[1], yy[2], yy[3], yy[4]), axis=0)
    z3 = np.zeros(35)
    dx = 0.5
    dy = 0.5
    dz = building_heights
    cmap = cm.get_cmap('OrRd')
    max_height = np.max(dz)  # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k - min_height) / max_height) for k in dz]
    ax1.bar3d(x3, y3, z3, dx, dy, dz, color=rgba)
    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    plt.savefig('figures/3dplot.pdf')

    # building heights histogram
    fig = plt.figure()
    count, bins, _ = plt.hist(building_heights, 35, density=True)
    fit = a * m ** a / bins ** (a + 1)
    plt.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
    plt.xlabel('Buildings height')
    plt.ylabel('Percent')
    plt.savefig('figures/hist.pdf')

    plt.show()
