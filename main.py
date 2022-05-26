import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM
import knot_class as kc
from knots_optimization import *

if __name__ == '__main__':
    knot_optimization = 1
    if knot_optimization:
        def knot_optimization():
            # def test_visual():

            coeffMod = [1.51, -5.06, 7.23, -2.03, -3.97]
            # ['1.53', '-5.04', '7.19', '-1.97', '-3.99']
            coeffStand = [1.715, -5.662, 6.381, -2.305, -4.356]
            coeffTest = [1.56, -5.11, 8.29, -2.37, -5.36]  # this is the best, I think MINE PAPER
            coeffTest = [1.39, -4.42, 7.74, -3.06, -4.08]  # dima 12 dots
            coeffTest = [1.371, -4.1911, 7.9556, -3.4812, -4.2231]  # 12 dots best [1.48, -4.78, 7.15, -2.25, -3.22]
            coeffTest = [1.828, -5.977, 6.577, -2.347, -3.488]  # w=1.3
            coeffTest = [1.7138467013697232, -5.304459582637312, 6.659112412081649, -2.5074137370635765, -4.407259374450308]
            # coeffTest = [1.35, -4.9, 7.43, -2.49, -3.1]
            # coeffTest = [1.41, -3.71, 7.44, -2.09, -4.26]  # dima 6 LAST
            # coeffTest = [1.26, -3.74, 7.71, -2.07, -4.25] # dima 5 dots BEST
            # посмотреть новые для 6 теста [1.41, -3.85, 7.28, -1.95, -4.25]
            # coeffTest /= np.sqrt(sum([a ** 2 for a in coeffTest])) * 0.1
            # coeffTest = np.array(coeffTest) * 1.51 / 1.5308532
            # Mod/Stand=0.95284, Mod/Test=1.03021 (i0=0.05)[1.41, -3.71, 7.44, -2.09, -4.26]
            i0 = 0.01
            iMin = i0 / 100
            xyMinMax = 4
            zMinMax = 1.1  # 2.6
            zRes = 121
            xRes = yRes = 101
            xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)

            # perfect from the paper
            # check_knot_paper(xyzMesh, coeffMod, deltaCoeff=[0.3] * 5, iMin=iMin, i0=i0, radiustest=0.05, steps=1000)
            plot_test = False
            if plot_test:
                xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
                fieldTest = fOAM.trefoil_mod(
                    *xyzMesh, w=1.3, width=1.2, k0=1, z0=0.,
                    coeff=coeffTest, coeffPrint=False
                )
                # fg.plot_3D_density(np.angle(fieldTest))
                fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 2] ** 2, axis_equal=True)
                fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2], axis_equal=True)
                fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
                if 1:
                    for i in range(zRes//2, zRes):
                        fOAM.plot_knot_dots(fieldTest[:, :, i], axesAll=False)
                plt.show()
                exit()
            check_knot_mine(xyzMesh, coeffTest, deltaCoeff=[0.3] * 5, steps=5000,
                            six_dots=True, testvisual=True,
                            circletest=True, radiustest=0.02,  # # # # # # # # # ## #
                            checkboundaries=True, boundaryValue=0.1,
                            xyzMeshPlot=fg.create_mesh_XYZ(xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
                                                           71, 71, 81, zMin=None))
            exit()

            fieldMod = fOAM.trefoil_mod(
                *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeffMod
                , coeffPrint=False,
            )

            fieldStand = fOAM.trefoil_mod(
                *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeffStand
                , coeffPrint=False,
            )
            fieldTest = fOAM.trefoil_mod(
                *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeffTest, coeffPrint=False
            )

            costMod = cost_function_paper(fieldMod[:, :, :], iMin=iMin, i0=i0)
            costStand = cost_function_paper(fieldStand[:, :, :], iMin=iMin, i0=i0)
            costTest = cost_function_paper(fieldTest[:, :, :], iMin=iMin, i0=i0)
            print(f"Mod: {costMod:.5f}, Stand:{costStand:.5f}, Test: {costTest:.5f}\n"
                  f"Mod/Stand={costMod / costStand:.5f}, Mod/Test={costMod / costTest:.5f}"
                  f" (i0={i0})")
            # exit()
            if 0:
                field = fieldTest
                fg.plot_2D(np.abs(field)[:, :, np.shape(field)[2] // 2] ** 2)
                fg.plot_2D(np.angle(field)[:, :, np.shape(field)[2] // 2])
                fOAM.plot_knot_dots(field, axesAll=False)
                plt.show()
                exit()
            initialCoeff = coeffTest
            field = fieldTest
            initialSum = cost_function_paper(field, iMin=iMin, i0=i0)

            print(initialCoeff)
            exit()

            # exit()

            # fg.plot_3D_density(np.angle(fieldTest))
            # fOAM.plot_knot_dots(fieldTest)
            # plt.show()
            # exit()
            #
            # print((np.abs(fieldMod[:, :, np.shape(fieldStand)[2]//2])).min(),
            #       (np.abs(fieldStand[:, :, np.shape(fieldStand)[2]//2])).min())
            # fg.plot_2D(np.abs(fieldMod)[:, :, np.shape(fieldMod)[2]//2 + 10] ** 2)
            # fg.plot_2D(np.angle(fieldMod)[:, :, np.shape(fieldMod)[2] // 2 + 10])
            # fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2]//2] ** 2)
            # fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2])
            # plt.show()

            # fOAM.plot_knot_dots(fieldStand)
            # fOAM.plot_knot_dots(fieldMod)
            # exit()
            # fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 4 * 3])
            # fg.plot_2D(np.abs(fieldTest)[:, :, -1])
            # fg.plot_2D(np.abs(fieldTest)[:, :, 0])
            # fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 4 * 3])
            # fg.plot_2D(np.angle(fieldTest)[:, :, -1])
            # fg.plot_2D(np.angle(fieldTest)[:, :, 0])
            # a0: 1.715, a1: -5.662, a2: 6.381, a3: -2.305, a4: -4.356,
            # [1.51, -5.06, 7.23, -2.03, -3.97]
            # [ 1.46 -5.11  7.28 -1.98 -4.02] 3.0 0.003
            # [ 1.51 -5.11  7.28 -1.98 -4.02] 3.75 0.003
            # [ 1.56 -5.11  7.28 -1.98 -3.92] 3.75 0.01
            # [ 1.56 -5.11  7.28 -1.98 -4.02] 3.75 0.00001
            deltaCoeff = [0.1025, 0.301, 0.4245, 0.1375, 0.193]
            dotsNumber = [3, 3, 3, 3, 3]
            i0 = 0.01
            # aInitial = [(a + b) / 2 for a, b in zip(coeffStand, coeffMod)]
            aInitial = [(a + b) / 2 for a, b in zip(coeffStand, coeffMod)]
            aAll = knot_permutations_all(aInitial, deltaCoeff, dotsNumber)
            sumArray = []
            for i, a in enumerate(aAll):
                print(i, len(aAll))
                field = fOAM.trefoil_mod(
                    *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=a, coeffPrint=False)
                # field = field / np.abs(field).max()
                sumArray.append(cost_function_paper(field, iMin=iMin, i0=i0))
            aBestIndex = sumArray.index(min(sumArray))
            aBest = aAll[aBestIndex]
            print(aBest, )
            if 0:
                coeff = []
                fg.plot_2D(np.angle(fOAM.trefoil_mod(
                    xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=[1.715, -5.662, 6.381, -2.305, -4.356], coeffPrint=False,
                )[:, :, 15]))
                fg.plot_2D(np.angle(fOAM.trefoil_mod(
                    xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=coeff, coeffPrint=False,
                )[:, :, 15]))
                plt.show()
            exit()
            # 6443266.7791220145
            # fg.plot_3D_density(np.abs(field))
            # exit()
            # fOAM.plot_knot_dots(field)
            # plt.show()


        knot_optimization()

    # not finished
    propagation_modification = False
    if propagation_modification:
        fhl.propagation_modification()

    milnor_research = False
    if milnor_research:
        fhl.milnor_research()

    propagation = False
    if propagation:
        # A = fg.readingFile('all_other_data/trefoil_exp_field.mat', fieldToRead='Uz', printV=False)
        A = fg.readingFile('all_other_data/trefoil_exp_field.mat', fieldToRead='Uz', printV=False)
        fg.plot_2D(np.angle(A[:, :, np.shape(A)[2] // 2]))
        fg.plot_2D(np.angle(A[:, :, np.shape(A)[2] // 2 + 3]))
        # fOAM.plot_knot_dots(A, bigSingularity=0, axesAll=0, cbrt=1, size=100, color='k')
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

        fOAM.plot_knot_dots(aProp, bigSingularity=0, axesAll=0, cbrt=1, size=100, color='k')

        plt.show()
        # knot_1_plane_propagation()
        # #

    knot_from_math = False
    if knot_from_math:
        fhl.knot_from_math_f()

    creating_table_knots = 0  # making_table1

    if creating_table_knots:
        SR = '0.95'
        knot = 'Trefoil'
        w = '1.2 12dots'  # Dima Cmex-
        # directoryName = (f'C:\\Users\\Cmex-\Box\\Knots Exp\\New_Data\\'
        #                  f'SR = {SR} (new)\\{knot}\\w = {w}/')
        directoryName = (
            f'C:\\WORK\\CODES\\knots_OAM_functions'
            f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
        # directoryName = (
        #     f'C:\\SCIENCE\\programming\\Python\\gitHub\\knots_OAM_functions'
        #     f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
        tableName = f'{knot}, SR={SR}, w={w}'
        kc.creat_knot_table(directoryName, tableName, show=True, cut=0.35)

    # metasurface for Jiannan
    metasurface_Jiannan = False
    if metasurface_Jiannan:
        def discretization_helper(x, steps, min, max):
            steps += 1  # 8 [0, 1, 2, 3, 4, 5, 6, 7, 8] where 0 and 8 are the same numbers
            discrArray = np.linspace(min, max, steps)
            return np.arange(steps)[np.abs(discrArray - x).argmin()]


        def discretization_phase(field, steps, min, max) -> type(np.array(())):
            answer = np.copy(field)
            for i, x in enumerate(field):
                if not len(np.shape(x)):
                    answer[i] = discretization_helper(x, steps, min, max)
                else:
                    answer[i] = discretization_phase(x, steps, min, max)
            answer[answer == steps] = 0
            return answer


        A = fg.readingFile('trefoil_300x300um.mat', fieldToRead='z', printV=False)
        A2 = (discretization_phase(A, 8, min=A.min(), max=A.max()))
        fg.plot_2D(A)
        fg.plot_2D(A2)
        plt.show()
        exit()

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
            fOAM.plot_knot_dots(field[185:327, 185:327, :])
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
