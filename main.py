import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM
import knot_class as kc

from scipy.fft import fftn, ifftn, fftshift, ifftshift

if __name__ == '__main__':
    propagation_modification = True
    if propagation_modification:
        xyMinMax = 5
        xRes = yRes = 150
        xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0.5, xRes, yRes, 40)
        field = fOAM.knot_all(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)
        fg.plot_2D(np.abs(ifftshift(fftn(field[:, :, np.shape(field)[2]//2]))))
        plt.show()
        exit()
        fieldProp = fg.simple_propagator_3D(field[:, :, np.shape(field)[2] // 2],
                                            xArray=np.linspace(-xyMinMax, xyMinMax, xRes),
                                            yArray=np.linspace(-xyMinMax, xyMinMax, yRes),
                                            dz=0.01, zSteps=100)
        fg.plot_2D(np.abs(fieldProp[:, :, -1]))
        fg.plot_2D(np.angle(fieldProp[:, :, -1]))
        plt.show()
        exit()
        # field = fg.size_array_increase_3D(field)
        f = fg.interpolation_complex(field[:, :, np.shape(field)[2] // 2],
                                     np.linspace(-3, 3, 80), np.linspace(-3, 3, 80))
        x, y = fg.create_mesh_XY(3, 3, 120, 120)
        field_inter = f[0](x, y) + 1j * f[1](x, y)
        # f_spec = fg.interpolation_complex(ifftshift(fftn(field_inter)),
        #                                   np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
        # x, y = fg.create_mesh_XY(2, 2, 150, 150)
        # f_spec_inter = f_spec[0](x, y) + 1j * f_spec[1](x, y)
        #
        # fg.plot_2D(np.abs(f_spec_inter))
        fg.plot_2D(np.abs(field_inter))
        fg.plot_2D(np.abs(field[:, :, np.shape(field)[2] // 2]))
        # fg.plot_2D(np.abs(ifftshift(fftn(field_inter))))
        plt.show()
        exit()
        field_inter_prop = fg.simple_propagator_3D(field_inter, dz=0.2, zSteps=30)
        fg.plot_2D(np.abs(field_inter_prop[:, :, -1]))
        fg.plot_2D(np.angle(field_inter_prop[:, :, -1]))
        plt.show()

    milnor_research = False
    if milnor_research:
        xyzMesh = fg.create_mesh_XYZ(2, 2, 0.5, 40, 40, 40)
        field = fOAM.milnor_Pol_testing(xyzMesh[0], xyzMesh[1], xyzMesh[2], 2, 3)
        fg.plot_2D(np.abs(field[:, :, np.shape(field)[2] // 2]))
        fg.plot_2D(np.angle(field[:, :, np.shape(field)[2] // 2]))
        fhl.plot_knot_dots(field, color='k', size=80, axesAll=True)
        # field = fOAM.knot_all(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, knot='trefoil_mod')
        # fg.plot_3D_density(np.angle(field))
        # fg.plot_2D(np.angle(field[:, :, np.shape(field)[2] // 2]))
        plt.show()

    propagation = 0
    if propagation:
        # A = fg.readingFile('all_other_data/trefoil_exp_field.mat', fieldToRead='Uz', printV=False)
        A = fg.readingFile('all_other_data/trefoil_exp_field.mat', fieldToRead='Uz', printV=False)
        fg.plot_2D(np.angle(A[:, :, np.shape(A)[2] // 2]))
        fg.plot_2D(np.angle(A[:, :, np.shape(A)[2] // 2 + 3]))
        # fhl.plot_knot_dots(A, bigSingularity=0, axesAll=0, cbrt=1, size=100, color='k')
        plt.show()
        exit()
        # plt.show()
        # exit()
        # fg.plot_2D(np.angle(A)[:, :, 8])
        # fg.plot_2D(np.abs(A)[:, :, 8])
        for i in range(7, 30, 1):
            print(i)
            fg.plot_2D(np.angle(A[:, :, 0 * np.shape(A)[2] // 2 + i]))
            fg.plot_2D(np.abs(A[:, :, 0 * np.shape(A)[2] // 2 + i]), map='gray')
            plt.show()
            input()

        # Aprop = fg.one_plane_propagator(A, dz=10, stepsNumber=10, shapeWrong=3)
        aProp = fg.one_plane_propagator(A, dz=10, stepsNumber=10, shapeWrong=0)
        for i in range(0, 19, 19):
            fg.plot_2D(np.angle(aProp[:, :, np.shape(aProp)[2] // 2 + 3]))
            plt.show()
        exit()
        aProp = fg.size_array_increase_3D(aProp)

        fhl.plot_knot_dots(aProp, bigSingularity=0, axesAll=0, cbrt=1, size=100, color='k')

        plt.show()
        # knot_1_plane_propagation()
        # #

    knot_from_math = False
    if knot_from_math:
        fOAM.knot_field_plot_save(xyMax=6, zMax=1.5, xyRes=300, zRes=11, w=1.2, width=1.5, k0=1,
                                  knot='trefoil_mod', axis_equal=True,
                                  save=True, saveName='trefoil_math_01',
                                  plot=True, plotLayer=None)
        plt.show()
        exit()
        field1 = np.load('trefoil_math_01.npy')
        for i in range(0, 1000, 2):
            print(i)
            fg.plot_2D(np.angle(field1[:, :, np.shape(field1)[2] // 2 + i]))
            fg.plot_2D(np.abs(field1[:, :, np.shape(field1)[2] // 2 + i]), map='gray')
            plt.show()
            input()
        exit()
        fg.plot_3D_density(np.angle(field1))
        # fhl.plot_knot_dots(field1)
        plt.show()

    creating_table_knots = 0  # making_table1

    if creating_table_knots:
        SR = '0.95'
        knot = 'Trefoil'
        w = '1.2 mod 2.25'  # Dima Cmex-
        # directoryName = (f'C:\\Users\\Cmex-\Box\\Knots Exp\\New_Data\\'
        #                  f'SR = {SR} (new)\\{knot}\\w = {w}/')
        directoryName = (
            f'C:\\WORK\\CODES\\knots_OAM_functions'
            f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
        tableName = f'{knot}, SR={SR}, w={w}'
        kc.creat_knot_table(directoryName, tableName, show=True, cut=0.5)

    studying_3D_OAM = 0
    if studying_3D_OAM:
        print(1 / 4 - 1 / 4 - 7 / 4 - 7 / 4 + 1 / 2 + 0 - 1 / 2 + 3 / 2 + 0 + 0 - 3 / 2 + 1 / 2 + (
                3 / 4 - 1 / 2) + 1 / 4 + 1 / 4 - 1 / 4)
        xyzMesh = fg.create_mesh_XYZ(3, 3, 3, 251, 251, 251)
        fieldOAM = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2],
                                  l=1, p=0, width=1, k0=1, x0=0, y0=0)
        shape = np.shape(fieldOAM)
        fg.plot_2D(np.angle(fieldOAM[shape[0] // 2 - 1:shape[0] // 2 + 2,
                            shape[1] // 2 - 1:shape[1] // 2 + 2,
                            np.shape(fieldOAM)[2] // 2]), map='hsv')
        xM, yM, zM = np.array(shape) // 2


        def angles_differences(array):
            minIndex = np.argmin(array)
            massiveAnglesDif = [array[i + 1] - array[i] for i in
                                range(minIndex - len(array), minIndex, 1)]
            print(np.sum(massiveAnglesDif[:-1]), massiveAnglesDif[-1])
            return massiveAnglesDif


        print(np.angle(fieldOAM[xM - 1, yM - 1, zM - 1]))
        exit()
        massiveAngles = [np.angle(fieldOAM[x, y, z]) for [x, y, z] in
                         [[xM - 1, yM - 1, zM], [xM, yM - 1, zM], [xM + 1, yM - 1, zM],
                          [xM + 1, yM, zM],
                          [xM + 1, yM + 1, zM], [xM, yM + 1, zM], [xM - 1, yM + 1, zM],
                          [xM - 1, yM, zM]]]
        angles_differences(massiveAngles)
        print(massiveAngles)
        massiveAngles2 = [np.angle(fieldOAM[x, y, z]) for [x, y, z] in
                          [[xM - 1, yM, zM - 1], [xM - 1, yM, zM], [xM - 1, yM, zM + 1],
                           [xM, yM, zM + 1],
                           [xM + 1, yM, zM + 1], [xM + 1, yM, zM], [xM + 1, yM, zM - 1],
                           [xM, yM, zM - 1]]]
        print(massiveAngles2)
        angles_differences(massiveAngles2)
        massiveAngles3 = [np.angle(fieldOAM[x, y, z]) for [x, y, z] in
                          [[xM, yM - 1, zM - 1], [xM, yM - 1, zM], [xM, yM - 1, zM + 1],
                           [xM, yM, zM + 1],
                           [xM, yM + 1, zM + 1], [xM, yM + 1, zM], [xM, yM + 1, zM - 1],
                           [xM, yM, zM - 1]]]
        print(massiveAngles3)
        angles_differences(massiveAngles2)

        # print(np.angle(fieldOAM[shape[0] // 2 - 1:shape[0] // 2 + 2,
        #       shape[1] // 2 - 1:shape[1] // 2 + 2,
        #       shape[2] // 2 - 1:shape[2] // 2 + 0]))
        plt.show()

    test_efild = False
    if test_efild:

        field = np.load('field_test2.npy')
        values = (field[:, :, :])

        if 1:
            fhl.plot_knot_dots(field[185:327, 185:327, :])
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
