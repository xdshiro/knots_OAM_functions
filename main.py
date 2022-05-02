import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM

def saving_knot_field(xyMax=3, xyMaxLim=3, xrRes=50, w=1, width=1, k0=1, save=False, plot=True):
    xyMinMax = [-xyMaxLim, xyMaxLim, -xyMaxLim, xyMaxLim]
    xMin, xMax = -xyMax, xyMax
    yMin, yMax = -xyMax, xyMax
    zMin, zMax = -0.6, 0.6
    # w = 1.2
    # z1 = 0.5  # 25 how long is propagation
    # z0 = 0  # 3 where we start (z0 meters before the center of the waist
    # dz = 0.02
    xArray = np.linspace(xMin, xMax, xrRes)
    yArray = np.linspace(yMin, yMax, xrRes)
    zArray = np.linspace(zMin, zMax, 50)
    # xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    xyzMesh = np.array(np.meshgrid(xArray, yArray, zArray, indexing='ij'))
    # fieldInitial = (100 * np.exp(1j * np.angle(fOAM.actual_trefoil(xyMesh[0], xyMesh[1], 0, w=w, z0=z0)))
    #                 * fOAM.LG_simple(xyMesh[0], xyMesh[1], z=z0, l=0, p=0, width=rho0 * 1.5, k0=k0))
    # plot_fast_2D(np.abs(fieldInitial), xArray, yArray, xyminmax=xyMinMax)
    # plot_fast_2D(np.angle(fieldInitial), xArray, yArray, xyminmax=xyMinMax)
    field = fOAM.actual_trefoil(xyzMesh[0], xyzMesh[1], z=xyzMesh[2], w=w, width=width, k0=k0)
    if plot:
        fg.plot_2D(np.abs(field), xArray, yArray)
        fg.plot_2D(np.angle(field), xArray, yArray, map='hsv')
    if save:
        np.save('C:/WORK/CODES/Knot Paper Final/field_test2', fOAM.actual_trefoil(
            xyzMesh[0], xyzMesh[1], z=xyzMesh[2], w=w, width=width, k0=k0))

if __name__ == '__main__':
    field1 = np.load('field_test2.npy')
    fieldTurb = fg.readingFile('Efield_100_500_SR_1.000000e-02.mat', fieldToRead="Efield", printV=False)

    propagation = False
    if propagation:
        fieldProp = fg.simple_propagator_3D(field1[:, :, 25], dz=1, zSteps=50, n0=1, k0=1)
        # fg.plot_3D_density(np.angle(testProp[:, :, :]), resDecrease=[1, 1, 1])
        fhl.plot_knot_dots(fieldProp)
        plt.show()

    knot_from_math = True
    if knot_from_math:
        saving_knot_field(xyMax=3, xyMaxLim=3, xrRes=50, w=1, width=1, k0=1, save=False, plot=True)

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
