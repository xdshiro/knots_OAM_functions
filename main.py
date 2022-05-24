import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM
import knot_class as kc

if __name__ == '__main__':
    knot_optimization = 1
    if knot_optimization:
        def knot_optimization():
            def cost_function_paper(field, iMin=0.01, i0=0.01, norm=1e6):
                I0 = np.max(np.abs(field)) ** 2
                I0 *= i0
                IFlat = np.ndarray.flatten(np.abs(field) ** 2)
                cutParam = I0 / i0 * iMin
                IFlat[IFlat < cutParam] = cutParam
                IMin = [1 / min(x, I0) for x in IFlat]
                # IMin = [1 / max(x, I0) for x in IFlat]
                return np.sum(IMin) / norm

            def knot_permutations_all(aInitial, deltaCoeff, dotsNumber):
                aValues = []
                for i, a in enumerate(aInitial):
                    aArray = np.linspace(a - deltaCoeff[i], a + deltaCoeff[i], dotsNumber[i])
                    aValues.append(aArray)
                return fg.permutations_all(*aValues)

            def circle_test(field, radius, testValue=1.):
                shape = np.shape(field)
                radius *= np.sqrt((shape[0] // 2 + shape[1] // 2) ** 2)
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        if np.sqrt((x - shape[0] // 2) ** 2 + (y - shape[1] // 2) ** 2) <= radius:
                            if np.abs(field[x, y]) > testValue:
                                return False
                return True

            def check_knot_paper(xyzMesh, coeff, deltaCoeff, iMin, i0, radiustest=0.05, steps=1000, ):
                field = fOAM.trefoil_mod(
                    *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=coeff
                    , coeffPrint=False,
                )
                sum = cost_function_paper(field, iMin=iMin, i0=i0)
                for i in range(steps):
                    print(i)
                    newCoeff = fg.random_list(coeff, deltaCoeff)
                    newField = fOAM.trefoil_mod(
                        *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                        coeff=newCoeff, coeffPrint=False)
                    newSum = cost_function_paper(newField, iMin=iMin, i0=i0)
                    if newSum < sum:
                        if circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
                                       radius=radiustest, testValue=2.5):
                            print(f'{sum / newSum: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                            fOAM.plot_knot_dots(newField, axesAll=False)
                            plt.show()
                            while True:
                                x = int(input())
                                if x == 9 or x == 1:
                                    sum = newSum
                                    coeff = newCoeff
                                    break
                                if x == 0:
                                    break
                        else:
                            print(f'It is not a knot anymore!')
                print(coeff)
                return coeff

                # check_knot_paper()
                # newCoeff = fg.random_list(initialCoeff, deltaCoeff)
                # newField = fOAM.trefoil_mod(
                #     *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                #     coeff=newCoeff, coeffPrint=False)
                # newSum = cost_function_paper(newField, iMin=iMin, i0=i0)
                # if newSum < initialSum:
                #     if circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
                #                    radius=0.04, testValue=2.5):
                #         print(f'{costMod / newSum: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                #         fOAM.plot_knot_dots(newField, axesAll=False)
                #         plt.show()
                #         while True:
                #             x = int(input())
                #             if x == 9 or x == 1:
                #                 initialSum = newSum
                #                 initialCoeff = newCoeff
                #                 break
                #             if x == 0:
                #                 break
                #
                #     else:
                #         print(f'It is not a knot anymore!')

            def check_knot_mine(xyzMesh, coeff, deltaCoeff, steps=1000, six_dots=True):
                field = fOAM.trefoil_mod(
                    *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=coeff
                    , coeffPrint=False,
                )

                def return_min_helper(dots, minDistance):

                    for i in range(len(dots) - 1):
                        for j in range(i + 1, len(dots)):
                            currentDistance = fg.distance_between_points(dots[i], dots[j])
                            # print(currentDistance)
                            if currentDistance < minDistance:
                                minDistance = currentDistance
                    return minDistance

                def dots12_check(dotsWithOAM, minDistance):
                    dotsPlus = [dot for dot, OAM in dotsWithOAM if OAM > 0]
                    dotsMinus = [dot for dot, OAM in dotsWithOAM if OAM < 0]
                    minDistancePlus = return_min_helper(dotsPlus, minDistance)
                    minDistanceMinus = return_min_helper(dotsMinus, minDistance)
                    if minDistanceMinus < minDistancePlus:
                        return minDistanceMinus
                    else:
                        return minDistancePlus

                def min_distance(field, six_dots=True):
                    fieldFull, dotsOnly = fg.cut_non_oam(np.angle(field[:, :, :]),
                                                         axesAll=False)
                    minDistance = float('inf')
                    have_seen_12_dots = False
                    for z in range(zRes):

                        dotsInZwithOam = [(list(dots[:2]), OAM) for (dots, OAM) in dotsOnly.items()
                                          if dots[2] == z]
                        dotsInZ = [dot for dot, OAM in dotsInZwithOam]
                        if (six_dots and len(dotsInZ) != 6) or (have_seen_12_dots and len(dotsInZ) != 12):
                            break
                        elif len(dotsInZ) == 12:  # 12 dots
                            have_seen_12_dots = True
                            minDistance = dots12_check(dotsInZwithOam, minDistance)

                        else:  # just 6 dots
                            minDistance = return_min_helper(dotsInZ, minDistance)
                        # fg.plot_2D(fieldFull[:, :, z])
                        # plt.show()
                    return minDistance
                    # print(fg.distance_between_points(dot, dotsInZ[0][:2]))

                minDistance = min_distance(field, six_dots=six_dots)
                for i in range(steps):
                    print(i)
                    newCoeff = fg.random_list(coeff, deltaCoeff)
                    newField = fOAM.trefoil_mod(
                        *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                        coeff=newCoeff
                        , coeffPrint=False,
                    )
                    minDistanceNew = min_distance(newField, six_dots=six_dots)
                    if minDistanceNew > minDistance:
                        print(f'{minDistance / minDistanceNew: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                        minDistance = minDistanceNew
                        coeff = newCoeff
                        # fOAM.plot_knot_dots(newField, axesAll=False)
                        # plt.show()
                        # while True:
                        #     x = int(input())
                        #     if x == 9 or x == 1:
                        #         minDistance = minDistanceNew
                        #         coeff = newCoeff
                        #         break
                        #     if x == 0:
                        #         break
                return coeff

            coeffMod = [1.51, -5.06, 7.23, -2.03, -3.97]
            # ['1.53', '-5.04', '7.19', '-1.97', '-3.99']
            coeffStand = [1.715, -5.662, 6.381, -2.305, -4.356]
            coeffTest = [1.56, -5.11, 8.29, -2.37, -5.36]  # [2.27, -6.0, 6.84, -0.84, -4.81] paper
            coeffTest = [1.41, -3.71, 7.44, -2.09, -4.26]  # dima 6 dots [1.69, -4.15, 7.64, -1.94, -4.93]
            coeffTest = [1.75, -3.65, 7.22, -1.77, -4.84]  # dima 12 dots
            coeffTest = [1.75, -4.65, 7.22, -1.77, -4.84]  # dima 12 dots w=1.3
            # посмотреть новые для 6 теста [1.41, -3.85, 7.28, -1.95, -4.25]
            # coeffTest /= np.sqrt(sum([a ** 2 for a in coeffTest])) * 0.1
            # coeffTest = np.array(coeffTest) * 1.51 / 1.5308532
            # Mod/Stand=0.95284, Mod/Test=1.03021 (i0=0.05)[1.41, -3.71, 7.44, -2.09, -4.26]
            i0 = 0.01
            iMin = i0 / 100
            xyMinMax = 4
            zMinMax = 1
            zRes = 71
            xRes = yRes = 101
            xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)

            # perfect from the paper
            # check_knot_paper(xyzMesh, coeffMod, deltaCoeff=[0.3] * 5, iMin=iMin, i0=i0, radiustest=0.05, steps=1000)
            plot_test = True
            if plot_test:
                xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
                fieldTest = fOAM.trefoil_mod(
                    *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
                    coeff=coeffTest, coeffPrint=False
                )
                fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 2] ** 2)
                fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2])
                fOAM.plot_knot_dots(fieldTest, axesAll=True)
                if 0:
                    for i in range(zRes):
                        fOAM.plot_knot_dots(fieldTest[:, :, i], axesAll=False)
                plt.show()
                exit()

            check_knot_mine(xyzMesh, coeffTest, deltaCoeff=[0.3] * 5, steps=1000, six_dots=True)
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
        w = '1.2 6dots'  # Dima Cmex-
        # directoryName = (f'C:\\Users\\Cmex-\Box\\Knots Exp\\New_Data\\'
        #                  f'SR = {SR} (new)\\{knot}\\w = {w}/')
        # directoryName = (
        #     f'C:\\WORK\\CODES\\knots_OAM_functions'
        #     f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
        directoryName = (
            f'C:\\SCIENCE\\programming\\Python\\gitHub\\knots_OAM_functions'
            f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
        tableName = f'{knot}, SR={SR}, w={w}'
        kc.creat_knot_table(directoryName, tableName, show=True, cut=0.4)

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
