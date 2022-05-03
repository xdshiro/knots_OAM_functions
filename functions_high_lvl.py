import functions_general as fg
import numpy as np
import matplotlib.pyplot as plt


# plotting dot's only from the Array of +-1
def plot_knot_dots(field, bigSingularity=False, axesAll=True, cbrt=True):
    dotsFull, dotsOnly = fg.cut_non_oam(np.angle(field),
                                        bigSingularity=bigSingularity, axesAll=axesAll, cbrt=cbrt)
    dotsPlus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == 1])
    dotsMinus = np.array([list(dots) for (dots, OAM) in dotsOnly.items() if OAM == -1])
    ax = fg.plot_scatter_3D(dotsPlus[:, 0], dotsPlus[:, 1], dotsPlus[:, 2])
    fg.plot_scatter_3D(dotsMinus[:, 0], dotsMinus[:, 1], dotsMinus[:, 2], ax=ax)
    plt.show()
