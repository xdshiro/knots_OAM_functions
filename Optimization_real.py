"""
This module optimize the knot in turbulence based on the region in N-D amplitudes space.
Finds the "knot-amplitude-space" and find the centre of it based on the weight coefficients
"""
import functions_OAM_knots as fOAM
import functions_general as fg
import matplotlib.pyplot as plt
import numpy as np
import timeit
import pyknotid.spacecurves as sp
import sympy
from python_tsp.distances import tsplib_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
We can make a graph for tsp, so it is not searching for all the dots, only close z
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
trefoilW16.fill_dotsList() this function is what makes everything slow 

Stop if the distance is too big. To find a hopf 
"""


class Singularities3D:
    """
    Work with singularities of any 3D complex field
    """

    def __init__(self, field3D=None):
        """
        :param field3D: any 3D complex field
        """
        self.field3D = field3D
        self.dotsDict = None  # self.dotsXY or self.dotsAll (can switch with self.swap()
        self.dotsXY = None  # singularities from XY planes
        self.dotsAll = None  # singularities from XY+XZ+YZ planes
        self.dotsList = None  # np.array [[x,y,z], [x,y,z], ...] random order
        self.mesh = None  # np.meshgrid from field_LG_combination
        self.coefficients = None  # [Cl1p1, Cl2p2...] from field_LG_combination
        self.modes = None  # [(l1,p1), (l2,p2) ...] from field_LG_combination
        # self.fill_dotsDict_from_field3D(_dotsXY=True)

    def field_LG_combination(self, mesh, coefficients, modes, **kwargs):
        """
        creating the field of any combination of LG beams
        Sum(Cl1p1 * LG_simple(*mesh, l=l1, p=p1, **kwargs))
        :param mesh: np.meshgrid
        :param coefficients: [Cl1p1, Cl2p2...] ...
        :param modes: [(l1,p1), (l2,p2) ...]
        """
        field = 0
        self.mesh = mesh
        self.coefficients = coefficients
        self.modes = modes
        for num, coefficient in enumerate(coefficients):
            field += coefficient * fOAM.LG_simple(*mesh, l=modes[num][0], p=modes[num][1], **kwargs)
        self.field3D = field

    def plot_plane_2D(self, zPlane, **kwargs):
        """
        Plot any z plane, both abs and angle
        :param zPlane: number of the plane to plot (0<=z<=shape[2])
        :return: None
        """
        fg.plot_2D(np.abs(self.field3D[:, :, zPlane]), **kwargs)
        fg.plot_2D(np.angle(self.field3D[:, :, zPlane]), **kwargs)
        plt.show()

    def plot_center_2D(self, **kwargs):
        """
        Plot the center plane (z=0 if from z is from -l to l)
        :return: None
        """
        shape = np.shape(self.field3D)
        self.plot_plane_2D(shape[2] // 2, **kwargs)

    def plot_dots(self, show=True, **kwargs):
        """
        Plot self.dots (scatter) using fOAM.plot_knot_dots()
        if self.dots is not initialized, initialization with self.fill_dotsDict_from_field3D()
        :param kwargs: Everything for fOAM.plot_knot_dots()
         also for self.fill_dotsDict_from_field3D()
        :return: None
        """
        if self.dotsDict is None:
            self.fill_dotsDict_from_field3D(**kwargs)
        ax = fOAM.plot_knot_dots(self.dotsDict, **kwargs)
        if show:
            plt.show()
        return ax
        # fg.distance_between_points()

    def plot_density(self, **kwargs):
        """
        Plot density on the browser
        :kwargs: Everything for fOAM.plot_knot_dots()
        :return: None
        """
        fg.plot_3D_density(np.angle(self.field3D), **kwargs)

    def fill_dotsDict_from_field3D(self, _dotsXY=True, **kwargs):
        """
        Filing self.dots with self.dotsXY. for self.dotsALL use parameter _dotsXY
        :param kwargs: Everything for fg.cut_non_oam()
        :param _dotsXY: if True, we are filling with self.dotsXY, otherwise with self.dotsALL
        :return: number of dots in self.dots
        """
        if _dotsXY:
            if self.dotsXY is None:
                self.fill_dotsXY(**kwargs)
            self.dotsDict = self.dotsXY
        else:
            if self.dotsAll is None:
                self.fill_dotsAll(**kwargs)
            self.dotsDict = self.dotsAll
        return len(self.dotsDict)

    def fill_dotsList(self):
        self.dotsList = np.array([[x, y, z] for (x, y, z) in self.dotsDict])

    def fill_dotsXY(self, **kwargs):
        """
        fill in self.dotsXY with using only XY cross-sections for singularities
        :param kwargs: fg.cut_non_oam besides axesAll
        :return:
        """
        garbage, self.dotsXY = fg.cut_non_oam(np.angle(self.field3D), axesAll=False, **kwargs)
        self.dotsDict = self.dotsXY

    def fill_dotsAll(self, **kwargs):
        """
        fill in self.dotsALL with using ALL 3 cross-sections for singularities
        :param kwargs: fg.cut_non_oam besides axesAll
        :return:
        """
        garbage, self.dotsAll = fg.cut_non_oam(np.angle(self.field3D), axesAll=True, **kwargs)
        self.dotsDict = self.dotsAll

    def dots_swap(self, **kwargs):
        """
        change self.dots between self.dotsXY and self.dotsAll
        if self.dots is not either of those -> self.fill_dotsDict_from_field3D()
        print the new self.dots
        :return: None
        """
        if self.dotsDict is self.dotsXY:
            if self.dotsAll is None:
                self.fill_dotsAll()
            self.dotsDict = self.dotsAll
            print(f'Dots are now in all 3 planes')
        elif self.dotsDict is self.dotsAll:
            if self.dotsXY is None:
                self.fill_dotsXY()
            self.dotsDict = self.dotsXY
            print(f'Dots are now in the XY-plane')
        else:
            self.fill_dotsDict_from_field3D(*kwargs)
            print(f'Dots were not dotsXY or dotsAll. Now dots are in the XY-plane')

    def boundary_step_test(self, coeffNum, step, funcCheck, **kwargs):
        if self.coefficients is None:
            print(f'No coefficients in the field are given')
            return 0
        coefficientsTest = np.copy(self.coefficients)
        coefficientsTest[coeffNum] += step
        fieldTest = Singularities3D()
        fieldTest.field_LG_combination(self.mesh, coefficientsTest, self.modes, **kwargs)
        fieldTest.plot_dots()


class Knot(Singularities3D):
    """
    Knot field (unknots also are knots in that case)
    """

    def __init__(self, field3D=None):
        """
        :param field3D: any 3D complex field
        """
        Singularities3D.__init__(self, field3D)
        self.dotsKnotList = None  # the actual knot (ordered line)
        self.knotSP = None

    def build_knot_pyknotid(self, **kwargs):
        """
        function build normilized pyknotid knot
        :return:
        """
        if self.dotsKnotList is None:
            self.fill_dotsKnotList()
        zMid = (max(z for x, y, z in self.dotsKnotList) + min(z for x, y, z in self.dotsKnotList)) / 2
        xMid = (max(x for x, y, z in self.dotsKnotList) + min(x for x, y, z in self.dotsKnotList)) / 2
        yMid = (max(y for x, y, z in self.dotsKnotList) + min(y for x, y, z in self.dotsKnotList)) / 2
        self.knotSP = sp.Knot(np.array(self.dotsKnotList) - [xMid, yMid, zMid], add_closure=False, **kwargs)

    def plot_knot(self, **kwargs):
        """
        plot the knot
        """
        if self.dotsKnotList is None:
            self.fill_dotsKnotList()
        if self.knotSP is None:
            self.build_knot_pyknotid(**kwargs)
        plt.plot([1], [1])
        self.knotSP.plot()
        plt.show()

    def fill_dotsKnotList(self):
        if self.dotsList is None:
            self.fill_dotsList()
        distance_matrix = euclidean_distance_matrix(self.dotsList, self.dotsList)
        permutation, distance = solve_tsp_local_search(distance_matrix)
        # print(dots[permutation])
        # print(permutation)
        self.dotsKnotList = self.dotsList[permutation]

    def fill_dotsKnotList_mine(self):
        """
        fill in self.dotsList by removing charge sign and placing everything into the list [[x, y, z], [x, y, z]...]
        :return: None
        """

        def min_dist(dot, dots):
            elements = [(fg.distance_between_points(dot, d), i) for i, d in enumerate(dots)]
            minEl = min(elements, key=lambda i: i[0])
            return minEl

        self.dotsKnotList = []
        dotsDict = {}
        for [x, y, z] in self.dotsDict:
            if not (z in dotsDict):
                dotsDict[z] = []
            dotsDict[z].append([x, y])
        indZ = next(iter(dotsDict))  # z coordinate
        indInZ = 0  # dot number in XY plane at z
        indexes = np.array([-1, 0, 1])  # which layers we are looking at
        currentDot = dotsDict[indZ].pop(indInZ)
        # distCheck = 20
        while dotsDict:
            # print(indZ, currentDot, dotsDict)
            minList = []  # [min, layer, position in Layer] for all indexes + indZ layers
            for i in indexes + indZ:  # searching the closest element among indexes + indZ
                if not (i in dotsDict):
                    continue
                minVal, min1Ind = min_dist(currentDot, dotsDict[i])
                # if minVal <= distCheck:
                minList.append([minVal, i, min1Ind])
            if not minList:
                newPlane = 2
                while not minList:
                    for i in [-newPlane, newPlane] + indZ:  # searching the closest element among indexes + indZ
                        if not (i in dotsDict):
                            continue
                        minVal, min1Ind = min_dist(currentDot, dotsDict[i])
                        # if minVal <= distCheck:
                        minList.append([minVal, i, min1Ind])
                    newPlane += 1
                if newPlane > 3:
                    print(f'we have some dots left. Stopped')
                    print(indZ, currentDot, dotsDict)
                    break
                print(f'dots are still there, the knot builred cannot use them all\nnew plane: {newPlane}')
            minFin = min(minList, key=lambda i: i[0])
            # if minFin[1] != indZ:
            self.dotsKnotList.append([*dotsDict[minFin[1]].pop(minFin[2]), minFin[1]])
            currentDot = self.dotsKnotList[-1][:-1]  # changing the current dot to a new one
            indZ = minFin[1]
            # else:
            #     dotsDict[minFin[1]].pop(minFin[2])
            #     currentDot = self.dotsList[-1][:-1]  # changing the current dot to a new one
            #     indZ = minFin[1]
            # currentDot = self.dotsList[-1][:-1][:]  # changing the current dot to a new one
            # indZ = minFin[1]
            # dotsDict[minFin[1]].pop(minFin[2])
            if not dotsDict[indZ]:  # removing the empty plane (0 dots left)
                del dotsDict[indZ]

    def check_knot_alex(self) -> bool:
        checkVal = None
        if self.knotSP is None:
            self.build_knot_pyknotid()
        t = sympy.symbols("t")
        self.alexPol = self.knotSP.alexander_polynomial(variable=t)
        if self.__class__.__name__ == 'Trefoil':
            checkVal = -t ** 2 + t - 1
        if checkVal is None:
            print(f'There is no check value for this type of knots')
            return False
        if self.alexPol == checkVal:
            return True
        return False

    """
        def fill_dotsKnotList(self):

        dotsDict = {}
        for [x, y, z] in self.dots:
            if not (z in dotsDict):
                dotsDict[z] = []
            dotsDict[z].append([x, y])
        indZ = next(iter(dotsDict))  # z coordinate
        indInZ = 0  # dot number in XY plane at z
        indexes = np.array([-1, 0, 1])  # which layers we are looking at
        currentDot = dotsDict[indZ].pop(indInZ)
        # distCheck = 10
        while dotsDict:
            # print(indInZ, currentDot, dotsDict)
            minList = []  # [min, layer, position in Layer] for all indexes + indZ layers
            for i in indexes + indZ:  # searching the closest element among indexes + indZ
                if not (i in dotsDict):
                    continue
                minVal, min1Ind = min_dist(currentDot, dotsDict[i])
                # if i == indZ:
                #     minVal /= 1.2
                # if minVal <= distCheck:
                minList.append([minVal, i, min1Ind])
            if not minList:
                print(f'dots are still there, the knot builred cannot use them all')
                break
            minFin = min(minList, key=lambda i: i[0])
            self.dotsList.append([*dotsDict[minFin[1]].pop(minFin[2]), minFin[1]])
            currentDot = self.dotsList[-1][:-1]  # changing the current dot to a new one
            indZ = minFin[1]
            if not dotsDict[indZ]:  # removing the empty plane (0 dots left)
                del dotsDict[indZ]

    """


class Trefoil(Knot):
    """
    Trefoil
    """

    def __init__(self, field3D=None):
        """
        initialization of the field. if field3D is none, initialization with a standard trefoil in
        the low resolution
        :param field3D: any 3D complex field
        """
        if field3D is None:
            xyMinMax = 2
            zMinMax = 0.8
            zRes = 80
            xRes = yRes = 80
            xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
            field3D = fOAM.trefoil_mod(*xyzMesh, w=1.7, width=1, k0=1, z0=0., aCoeff=None, coeffPrint=False)
        Knot.__init__(self, field3D)


if __name__ == '__main__':
    def func_time_main():
        def check_simple_sphere(singular: Singularities3D, xyzMesh, dotCenter=(0, 0, 0), radiusOuter=None,
                                radiusInner=None):
            if singular.dotsDict is None:
                singular.fill_dotsXY()
            if len(singular.dotsDict) < 5:
                return False
            if radiusInner is not None:
                for dot in singular.dotsDict:
                    dotXYZ = [xyzMesh[i][dot] for i in range(len(dot))]
                    if fg.distance_between_points(dotXYZ, dotCenter) < radiusInner:
                        return False
            if radiusOuter is not None:
                for dot in singular.dotsDict:
                    dotXYZ = [xyzMesh[i][dot] for i in range(len(dot))]
                    if fg.distance_between_points(dotXYZ, dotCenter) > radiusOuter:
                        return False
            return True

        def check_one_direction(check_func):
            xMinMax = 1
            yMinMax = 0.8
            xMin = -0.4
            zMinMax = 0.35
            zRes = 20
            xRes = yRes = 20
            unknot = Singularities3D()
            xyzMesh = fg.create_mesh_XYZ(xMinMax, yMinMax, zMinMax, xRes, yRes, zRes, xMin=xMin, zMin=None)

            # print(unknot.dotsDict, unknot.dotsXY)
            # unknot.plot_dots()
            dotCenter = ((xMinMax + xMin) / 2, 0, 0)
            point = [8, -9.5, -5]
            pointB = [0, 0, 0]
            ind = 0
            min1 = point[ind]
            h1 = -0.05
            for i in range(10):
                point[ind] += h1
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    h1 *= 2
                else:
                    break
            max1 = point[ind]
            for i in range(10):
                point[ind] = (min1 + max1) / 2
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    min1 = point[ind]
                else:
                    max1 = point[ind]
            pointB[ind] = (min1 + max1) / 2
            print(pointB)
            point = [8, -9.5, -5]

            ind = 1
            min1 = point[ind]
            for i in range(10):
                point[ind] += h1
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    h1 *= 2
                else:
                    break
            max1 = point[ind]
            for i in range(10):
                point[ind] = (min1 + max1) / 2
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    min1 = point[ind]
                else:
                    max1 = point[ind]
            pointB[ind] = (min1 + max1) / 2
            print(pointB)
            point = [8, -9.5, -5]
            ind = 2
            min1 = point[ind]
            for i in range(10):
                point[ind] += h1
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    h1 *= 2
                else:
                    break
            max1 = point[ind]
            for i in range(10):
                point[ind] = (min1 + max1) / 2
                unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                unknot.fill_dotsXY()
                check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                # print(check, point)
                if check:
                    min1 = point[ind]
                else:
                    max1 = point[ind]
            pointB[ind] = (min1 + max1) / 2
            print(pointB)
            point = [8, -9.5, -5]
            a00Ar = np.linspace(7.4693603515625, 8.5269775390625, 10)
            a01Ar = np.linspace(-10.857617187499999, -8.892382812499998, 10)
            a10Ar = np.linspace(-6.07265625, -3.998046875, 10)
            ans = []
            for a00 in a00Ar:
                for a01 in a01Ar:
                    for a10 in a10Ar:
                        point = [a00, a01, a10]
                        unknot.field_LG_combination(xyzMesh, point, [(0, 0), (0, 1), (1, 0)])
                        unknot.fill_dotsXY()
                        check = check_func(unknot, xyzMesh, dotCenter=dotCenter, radiusInner=0.40, radiusOuter=0.65)
                        #
                        # print(point)
                        if check:
                            ans.append(point)

            ans = np.array(ans)

            fg.plot_scatter_3D(ans[:, 0], ans[:, 1], ans[:, 2])
            plt.show()

        # check_one_direction(check_simple_sphere)

        xMinMax = 3
        yMinMax = 3
        zMinMax = 0.8
        zRes = 70
        xRes = yRes = 70
        xyzMesh = fg.create_mesh_XYZ(xMinMax, yMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
        beam = Singularities3D()
        coeff = [1.715, -5.662, 6.381 * 17/16, -2.305, -4.356]
        phase = [0, 0, np.pi / 32 * 0, 0, 0]
        coeff = [a * np.exp(1j*p) for a, p in zip(coeff, phase)]
        beam.field_LG_combination(xyzMesh, coeff, [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0)])
        # beam.plot_center_2D(axis_equal=True)
        beam.fill_dotsAll()
        beam.plot_dots()
        exit()

        trefoil = Knot()
        w = 1.6
        H = 1
        a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
        a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
        a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
        a03 = (-6 * w ** 6) / H ** 6
        a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
        trefoilCoeff = [a00, a01, a02, a03, a30]

        trefoil.field_LG_combination(xyzMesh, trefoilCoeff,
                                    [(0, 0), (0, 1), (0, 2), (0, 3), (3, 0)])
        # unknot.fill_dotsXY()
        trefoil.plot_center_2D(axis_equal=True)
        hopf = Knot()
        w = 1.475
        a00 = 1 - 2 * w ** 2 + 2 * w ** 4
        a01 = 2 * w ** 2 - 4 * w ** 4
        a02 = 2 * w ** 4
        a20 = 4 * np.sqrt(2) * w ** 2
        hopfCoeff = [a00, a01, a02, a20]
        xyzMesh = fg.create_mesh_XYZ(xMinMax, yMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
        hopf.field_LG_combination(xyzMesh, hopfCoeff,
                                     [(0, 0), (0, 1), (0, 2), (2, 0)])
        # unknot.fill_dotsXY()
        hopf.plot_center_2D()
        # print(unknot.dotsDict)
        # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # for dot in unknot.dotsDict:
        #     dotXYZ = (xyzMesh[i][dot] for i in range(len(dot)))
        #     # print(xyzMesh[0][dot], xyzMesh[1][dot], xyzMesh[2][dot])
        #     # fg.plot_scatter_3D(*dotXYZ, ax=ax)
        # # plt.show()
        exit()
        # unknot.plot_dots(show=True)
        # unknot.plot_dots(show=True)
        # plt.show()
        # unknot.plot_density()
        # plt.show()
        # unknot.plot_density()

        # Hopf.boundary_step_test(coeffNum=2, step=1, funcCheck=None)
        # trefoilW16.plot_knot()
        # trefoilW16.build_knot_pyknotid()
        # t = sympy.symbols("t")
        # print(trefoilW16.knotSP.alexander_polynomial(variable=t))


    def func_time1():
        trefoilW16 = Trefoil()
        # trefoilW16.fill_dotsKnotList_mine()
        print(trefoilW16.check_knot())
        # trefoilW16.fill_dotsList()
        # trefoilW16.build_knot_pyknotid()
        # trefoilW16.plot_knot()
        exit()


    def func_time2():
        trefoilW16 = Trefoil()

        trefoilW16.fill_dotsList()
        trefoilW16.fill_dotsKnotList_mine()
        trefoilW16.build_knot_pyknotid()
        trefoilW16.plot_knot()


    timeit.timeit(func_time_main, number=1)
    # trefoilW16.plot_dots()
    # trefoilW16.plot_center_2D()
    # trefoilW16.plot_density()
    # runs1 = timeit.timeit(func_time1, number=1)
    # runs2 = timeit.timeit(func_time2, number=1)
    # print(runs1, runs2)
