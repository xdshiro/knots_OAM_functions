import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# mine
import functions_general as fg
import functions_high_lvl as fhl

# field = fg.readingFile('trefoil_300x300um.mat', fieldToRead="z")
# fg.plot_2D(field)
# plt.show()
# exit()
if __name__ == '__main__':

    test_efild = True
    if test_efild:

        field = np.load('field_test2.npy')
        values = (field[:, :, :])
        """testProp = fg.simple_propagator_3D(values[:, :, 25], dz=0.5, zSteps=10, n0=1, k0=1)
        fg.plot_2D(np.abs(testProp[:, :, -1]))
        plt.show()
        exit()"""
        if 0:
            fieldTurb = fg.readingFile('Efield_100_500_SR_1.000000e-02.mat', fieldToRead="Efield", printV=False)
            fhl.plot_knot_dots(fieldTurb[185:327, 185:327, :])
            plt.show()
            exit()

        # test_filter = fg.cut_fourier_filter(np.exp(1j * np.angle(values)), radiusPix=13)
        test_filter = fg.cut_fourier_filter(np.exp(np.abs(values)), radiusPix=3)

        fg.plot_2D(np.log(np.abs(test_filter[:, :, 25])))
        plt.show()
        exit()
        fg.plot_2D(values[:, :, 25])
        #fg.plot_scatter_3D([1, 2, 3], [2, 3, 1], [1, 2, 3])
        #plt.show()
        plt.show()
        # fg.plot_3D_density(test, resDecrease=[1, 1, 1], opacity=0.9, surface_count=11,
        #                   opacityscale='extremes')
    # [[-1, 1], [-0.9, 0], [0.9, 0], [1, 1]]
    experiment_filed = False
    if experiment_filed:
        matFile = sio.loadmat('C:/Users/Dima/Box/Knots Exp/No Turbulence (exp)/trefoil_-3.500000e-01_3.mat',
                              appendmat=False)
        """knot = Knot3(dotsFileName="trefoil_exp_d.mat", dz=1, clean=1,
                     angleCheck=180, distCheck=5, layersStep=1)
        knot.plotDots()
        plt.show()"""
        zPos = 0
        field = np.array(matFile['frame'])
        shapeField = np.shape(field)
        print(shapeField)
        field2D = field[:, :, zPos]
        plotField = field2D
        i = -1
        for row in plotField:
            i += 1
            j = -1
            for dot in row:
                j += 1
                """if np.real(dot) == np.imag(dot)  and np.real(dot) == 0:
                    plotField[i, j] = 0
                else:
                    plotField[i, j] = 1"""
        x = range(shapeField[0])
        y = range(shapeField[1])

        fg.plot_2D(np.real(plotField), x, y)  # , xlim=[225, 275], ylim=[225, 275]
        plt.show()
        plt.close()
        exit()
