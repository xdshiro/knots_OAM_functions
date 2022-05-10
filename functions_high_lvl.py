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


# used for the comparison between different sizes of the phase screens for knots
def resizing_knot_test():
    field1 = np.load('trefoil_math_01.npy')
    # fg.plot_2D(np.abs(field1[:, :, np.shape(field1)[2] // 2 + 20]))
    # fg.plot_2D(np.angle(field1[:, :, np.shape(field1)[2] // 2 + 20]), map='hsv')
    # # fhl.plot_knot_dots(field1)
    # plt.show()
    # exit()
    stepNumber = 36
    fieldProp = fg.simple_propagator_3D(field1[:, :, np.shape(field1)[2] // 2],
                                        dz=2/4, zSteps=stepNumber, n0=1, k0=1)
    # fg.plot_3D_density(np.angle(testProp[:, :, :]), resDecrease=[1, 1, 1])
    fg.plot_2D(np.abs(fieldProp[:, :, stepNumber]))
    fg.plot_2D(np.angle(fieldProp[:, :, stepNumber]), map='hsv')
    plt.show()
    # plot_knot_dots(fieldProp)
    # plt.show()
