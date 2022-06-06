import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM

import winsound


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


def test_visual(dotsOnly, coeff=None, xyzMeshVisual=None, sound=True, knot='trefoil'):
    if coeff is None:
        dots = np.array([list(dots) for (dots, OAM) in dotsOnly.items()])
        fg.plot_scatter_3D(dots[:, 0], dots[:, 1], dots[:, 2])
        plt.show()
    else:
        if knot == 'trefoil':
            fieldTest = fOAM.trefoil_mod(
                *xyzMeshVisual, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeff, coeffPrint=False
            )
        else:
            fieldTest = fOAM.hopf_mod(
                *xyzMeshVisual, w=1.2, width=1.2, k0=1, z0=0.,
                coeff=coeff, coeffPrint=False
            )
        if sound:
            duration = 300  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, duration)
        fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
        plt.show()

    while True:
        x = int(input())
        if x == 9 or x == 1:
            return True
        if x == 0:
            return False


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


def min_distance(dotsOnly, zRes, six_dots=True):
    minDistance = float('inf')
    have_seen_12_dots = False
    for z in range(zRes):

        dotsInZwithOam = [(list(dots[:2]), OAM) for (dots, OAM) in dotsOnly.items()
                          if dots[2] == z]
        dotsInZ = [dot for dot, OAM in dotsInZwithOam]
        if (six_dots and len(dotsInZ) != 6) or (have_seen_12_dots and len(dotsInZ) != 12):
            break
        elif 6 < len(dotsInZ) < 12:
            continue
        elif len(dotsInZ) == 12:  # 12 dots
            if six_dots:
                return 0
            have_seen_12_dots = True
            minDistance = dots12_check(dotsInZwithOam, minDistance)

        else:  # just 6 dots
            potMinDistance = return_min_helper(dotsInZ, minDistance)
            if (not (potMinDistance < minDistance * 0.94)) or z == 0:  #######################
                minDistance = potMinDistance
            else:
                break
        # fg.plot_2D(fieldFull[:, :, z])
        # plt.show()
    return minDistance
    # print(fg.distance_between_points(dot, dotsInZ[0][:2]))


def min_distance_hopf(dotsOnly, zRes, six_dots=False):
    minDistance = float('inf')
    have_seen_8_dots = False
    for z in range(zRes):

        dotsInZwithOam = [(list(dots[:2]), OAM) for (dots, OAM) in dotsOnly.items()
                          if dots[2] == z]
        dotsInZ = [dot for dot, OAM in dotsInZwithOam if OAM > 0]
        if (six_dots and len(dotsInZ) != 4) or (have_seen_8_dots and len(dotsInZ) != 8):
            break
        elif 4 < len(dotsInZ) < 8:
            continue
        elif len(dotsInZ) == 8:  # 12 dots
            if six_dots:
                return 0
            have_seen_8_dots = True
            minDistance = dots12_check(dotsInZwithOam, minDistance)

        else:  # just 6 dots
            potMinDistance = return_min_helper(dotsInZ, minDistance)
            minDistance = potMinDistance
        # fg.plot_2D(fieldFull[:, :, z])
        # plt.show()
    return minDistance
    # print(fg.distance_between_points(dot, dotsInZ[0][:2]))


def empty_space_check(dotsOnly, zRes, valueTest):
    zArray = [dot[2] for dot in dotsOnly]
    zArray.sort()
    if max(zArray) < zRes * (1 - valueTest):
        return True
    else:
        return False
    # add here the distance in between


def check_knot_mine(xyzMesh, coeff, deltaCoeff, steps=1000, six_dots=True, checkboundaries=False, boundaryValue=0.2,
                    circletest=True, radiustest=0.05, testvisual=False, xyzMeshPlot=None):
    field = fOAM.trefoil_mod(
        *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
        coeff=coeff
        , coeffPrint=False,
    )
    xRes, yRes, zRes = np.shape(xyzMesh)[1:]
    dotsOnly = fg.cut_non_oam(np.angle(field[:, :, :]),
                              axesAll=False)[1]
    minDistance = min_distance(dotsOnly, zRes, six_dots=six_dots)
    for i in range(steps):
        newCoeff = fg.random_list(coeff, deltaCoeff)
        newField = fOAM.trefoil_mod(
            *xyzMesh, w=1.2, width=1.2, k0=1, z0=0.,
            coeff=newCoeff
            , coeffPrint=False,
        )
        dotsOnly = fg.cut_non_oam(np.angle(newField[:, :, :]),
                                  axesAll=False)[1]
        if circletest:
            if not circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
                               radius=radiustest, testValue=2.5):
                print('circle')
                continue
        if checkboundaries:
            if not empty_space_check(dotsOnly, zRes, boundaryValue):
                print('boundaries')
                continue
        minDistanceNew = min_distance(dotsOnly, zRes, six_dots=six_dots)
        print(i, f'{minDistance: .2f}', f'{minDistanceNew:.2f}', newCoeff)
        if minDistanceNew > minDistance:
            if testvisual:
                print('test visual')
                if test_visual(dotsOnly, coeff, xyzMeshPlot):
                    print(f'{minDistance / minDistanceNew: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                    minDistance = minDistanceNew
                    coeff = newCoeff
            else:
                print(f'{minDistance / minDistanceNew: .3f}', [float(f'{a:.2f}') for a in newCoeff])
                minDistance = minDistanceNew
                coeff = newCoeff

    return coeff


def check_knot_mine_hopf(xyzMesh, coeff, deltaCoeff, steps=1000, six_dots=True, checkboundaries=False,
                         boundaryValue=0.2, width=1.4,
                         circletest=True, radiustest=0.05, testvisual=False, xyzMeshPlot=None):
    field = fOAM.hopf_mod(
        *xyzMesh, w=1.4, width=width, k0=1, z0=0.,
        coeff=coeff
        , coeffPrint=False,
    )
    xRes, yRes, zRes = np.shape(xyzMesh)[1:]
    dotsOnly = fg.cut_non_oam(np.angle(field[:, :, :]),
                              axesAll=False)[1]
    minDistance = min_distance_hopf(dotsOnly, zRes, six_dots=six_dots)
    for i in range(steps):
        newCoeff = fg.random_list(coeff, deltaCoeff)
        newField = fOAM.hopf_mod(
            *xyzMesh, w=1.4, width=width, k0=1, z0=0.,
            coeff=newCoeff
            , coeffPrint=False,
        )
        dotsOnly = fg.cut_non_oam(np.angle(newField[:, :, :]),
                                  axesAll=False)[1]
        if circletest:
            if not circle_test(np.angle(newField)[:, :, np.shape(newField)[2] // 2],
                               radius=radiustest, testValue=2.5):
                print('circle')
                continue
        minDistanceNew = min_distance_hopf(dotsOnly, zRes, six_dots=six_dots)
        print(i, f'{minDistance: .2f}', f'{minDistanceNew:.2f}', newCoeff)
        if minDistanceNew > minDistance:
            if checkboundaries:
                fieldBound = fOAM.hopf_mod(
                    *xyzMeshPlot, w=1.4, width=width, k0=1, z0=0.,
                    coeff=coeff, coeffPrint=False
                )
                dotsOnlyBound = fg.cut_non_oam(np.angle(fieldBound[:, :, :]),
                                               axesAll=False)[1]
                xResB, yResB, zResB = np.shape(xyzMeshPlot)[1:]
                if not empty_space_check(dotsOnlyBound, zResB, boundaryValue):
                    print(f'Boundary is bad')
                    continue
                else:
                    print(f'Boundary is good')
            if testvisual:
                print('test visual')
                if not test_visual(dotsOnly, coeff, xyzMeshPlot, knot='hopf'):
                    print('visualy no')
                    continue
            print(f'{minDistance / minDistanceNew: .3f}', [float(f'{a:.2f}') for a in newCoeff])
            minDistance = minDistanceNew
            coeff = newCoeff

    return coeff