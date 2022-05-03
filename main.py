import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM
import knot_class as kc


if __name__ == '__main__':

    propagation = False
    if propagation:
        field1 = np.load('trefoil_math_01.npy')
        fieldTurb = fg.readingFile('Efield_100_500_SR_1.000000e-02.mat', fieldToRead="Efield", printV=False)

        fieldProp = fg.simple_propagator_3D(field1[:, :, np.shape(field1[2]//2)],
                                            dz=1, zSteps=50, n0=1, k0=1)
        # fg.plot_3D_density(np.angle(testProp[:, :, :]), resDecrease=[1, 1, 1])
        fhl.plot_knot_dots(fieldProp)
        plt.show()

    knot_from_math = False
    if knot_from_math:
        fOAM.knot_field_plot_save(xyMax=5, zMax=0.7, xyRes=273, zRes=51, w=1.4, width=1.4, k0=1,
                                  knot='trefoil',
                                  save=True, saveName='trefoil_math_01',
                                  plot=True, plotLayer=None)
        plt.show()

    creating_table_knots = True  # making_table1
    if creating_table_knots:
        SR = '0.001'
        knot = 'trefoil'
        w = '1.1'
        directoryName = f'C:\\Users\\Dima\\Box\\Knots Exp\\New_Data\\SR = {SR}\\{knot}\\w = {w}/'
        tableName = f'{knot}, SR={SR}, w={w}'
        kc.creat_knot_table(directoryName, tableName)






    test_efild = False
    if test_efild:

        field = np.load('field_test2.npy')
        values = (field[:, :, :])

        if 1:
            fhl.plot_knot_dots(fieldTurb[185:327, 185:327, :])
            plt.show()
            exit()

        # test_filter = fg.cut_fourier_filter(np.exp(1j * np.angle(values)), radiusPix=13)
        test_filter = fg.cut_fourier_filter(np.exp(np.abs(values)), radiusPix=3)

        fg.plot_2D(np.log(np.abs(test_filter[:, :, 25])))
        plt.show()
        exit()
        fg.plot_2D(values[:, :, 25])
        # fg.plot_scatter_3D([1, 2, 3], [2, 3, 1], [1, 2, 3])
        # plt.show()
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