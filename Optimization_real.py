"""
This module optimize the knot in turbulence based on the region in N-D amplitudes space.
Finds the "knot-amplitude-space" and find the centre of it based on the weight coefficients
"""
import my_modules.functions_OAM_knots as fOAM
import my_modules.functions_general as fg
import matplotlib.pyplot as plt
import numpy as np
import timeit


def min_dist(dot, dots):
    elements = [(fg.distance_between_points(dot, d), i) for i, d in enumerate(dots)]
    minEl = min(elements, key=lambda i: i[0])
    return minEl


class Singularities3D:
    """
    Work with singularities of any 3D complex field
    """

    def __init__(self, field3D):
        """
        :param field3D: any 3D complex field
        """
        self.field3D = field3D
        self.dots = None  # self.dotsXY or self.dotsAll (can switch with self.swap()
        self.dotsXY = None  # singularities from XY planes
        self.dotsAll = None  # singularities from XY+XZ+YZ planes
        self.create_dots_from_field3D(_dotsXY=True)

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

    def plot_density(self, **kwargs):
        """
        Plot density on the browser
        :kwargs: Everything for fOAM.plot_knot_dots()
        :return: None
        """
        fg.plot_3D_density(np.angle(self.field3D), **kwargs)
        plt.show()

    def create_dots_from_field3D(self, _dotsXY=True, **kwargs):
        """
        Filing self.dots with self.dotsXY. for self.dotsALL use parameter _dotsXY
        :param kwargs: Everything for fg.cut_non_oam()
        :param _dotsXY: if True, we are filling with self.dotsXY, otherwise with self.dotsALL
        :return: number of dots in self.dots
        """
        if _dotsXY:
            if self.dotsXY is None:
                self.fill_dotsXY(**kwargs)
            self.dots = self.dotsXY
        else:
            if self.dotsAll is None:
                self.fill_dotsAll(**kwargs)
            self.dots = self.dotsAll
        return len(self.dots)

    def fill_dotsXY(self, **kwargs):
        """
        fill in self.dotsXY with using only XY cross-sections for singularities
        :param kwargs: fg.cut_non_oam besides axesAll
        :return:
        """
        garbage, self.dotsXY = fg.cut_non_oam(np.angle(self.field3D), axesAll=False, **kwargs)

    def fill_dotsAll(self, **kwargs):
        """
        fill in self.dotsALL with using ALL 3 cross-sections for singularities
        :param kwargs: fg.cut_non_oam besides axesAll
        :return:
        """
        garbage, self.dotsAll = fg.cut_non_oam(np.angle(self.field3D), axesAll=True, **kwargs)

    def dots_swap(self, **kwargs):
        """
        change self.dots between self.dotsXY and self.dotsAll
        if self.dots is not either of those -> self.create_dots_from_field3D()
        print the new self.dots
        :return: None
        """
        if self.dots is self.dotsXY:
            if self.dotsAll is None:
                self.fill_dotsAll()
            self.dots = self.dotsAll
            print(f'Dots are now in all 3 planes')
        elif self.dots is self.dotsAll:
            if self.dotsXY is None:
                self.fill_dotsXY()
            self.dots = self.dotsXY
            print(f'Dots are now in the XY-plane')
        else:
            self.create_dots_from_field3D(*kwargs)
            print(f'Dots were not dotsXY or dotsAll. Now dots are in the XY-plane')

    def plot_dots(self, **kwargs):
        """
        Plot self.dots (scatter) using fOAM.plot_knot_dots()
        if self.dots is not initialized, initialization with self.create_dots_from_field3D()
        :param kwargs: Everything for fOAM.plot_knot_dots()
         also for self.create_dots_from_field3D()
        :return: None
        """
        if self.dots is None:
            self.create_dots_from_field3D(**kwargs)
        fOAM.plot_knot_dots(self.dots, **kwargs)



        # fg.distance_between_points()
    # def addClosest(self):
    #     tempIndex = -19
    #     tempDistance = self.distCheck * self.dz
    #     foundDot = False
    #     for i in range(len(self.dots)):
    #         if abs(self.dots[i][2] - self.knot[-1][2]) < (
    #                 self.layersStep + 1) * self.dz:  # and self.dots[i][2] - self.knot[-1][2] != 0:  #self.dots[i][2] != self.knot[-1][2] or self.dots[i][2] == self.knot[-1][2]
    #             distance = distancePoints2D(self.dots[i], self.knot[-1])  # distancePoints2D
    #             if distance < tempDistance:
    #                 tempDistance = distance
    #                 tempIndex = i
    #                 foundDot = True
    #     if foundDot:
    #         self.knot = np.append(self.knot, [self.dots[tempIndex]], axis=0)
    #         self.dots = np.delete(self.dots, tempIndex, axis=0)
    #         self.knotCurrentAngleCheck()
    #         return True
    #     else:
    #         print(f"\033[03m We haven't found a dot:\n\t\t current length: {len(self.knot)}, "
    #               f"total amount: {len(self.initialDots)} (before cleaning)")
    #         return False


class Knot(Singularities3D):
    """
    Knot field (unknots also are knots in that case)
    """

    def __init__(self, field3D=None):
        """
        :param field3D: any 3D complex field
        """
        super().__init__(field3D)
        self.dotsList = []  # the actual knot (ordered line)

    def plot_knot(self, **kwargs):
        """
        plot the knot
        """
        if not self.dotsList:
            self.fill_dotsList()
        fOAM.plot_knot_dots(self.dotsList, **kwargs)

    def fill_dotsList(self):
        """
        fill in self.dotsList by removing charge sign and placing everything into the list [[x, y, z], [x, y, z]...]
        :return: None
        """
        dotsDict = {}
        for [x, y, z] in self.dots:
            if not (z in dotsDict):
                dotsDict[z] = []
            dotsDict[z].append([x, y])
        indZ = next(iter(dotsDict))  # z coordinate
        indInZ = 0  # dot number in XY plane at z
        indexes = np.array([-1, 0, 1])  # which layers we are looking at
        currentDot = dotsDict[indZ].pop(indInZ)
        while dotsDict:
            minList = []  # [min, layer, position in Layer] for all indexes + indZ layers
            for i in indexes + indZ:  # searching the closest element among indexes + indZ
                if not (i in dotsDict):
                    continue
                minVal, min1Ind = min_dist(currentDot, dotsDict[i])
                minList.append([minVal, i, min1Ind])
            minFin = min(minList, key=lambda i: i[0])
            self.dotsList.append([minFin[1], *dotsDict[minFin[1]].pop(minFin[2])])
            currentDot = self.dotsList[-1][1:]  # changing the current dot to a new one
            indZ = minFin[1]
            if not dotsDict[indZ]:  # removing the empty plane (0 dots left)
                del dotsDict[indZ]


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
            xyMinMax = 4
            zMinMax = 0.6
            zRes = 40
            xRes = yRes = 90
            xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
            field3D = fOAM.trefoil_mod(*xyzMesh, w=1.4, width=1, k0=1, z0=0., coeff=None, coeffPrint=False)
        super().__init__(field3D)

    def trefoil_check_automated(self):
        print(self.dots)


if __name__ == '__main__':
    trefoilW16 = Trefoil()
    # trefoilW16.dots_swap()
    # trefoilW16.plot_center_2D()
    # trefoilW16.fill_dotsList()
    trefoilW16.plot_dots()
    # trefoilW16.plot_dots()
    # trefoilW16.plot_center_2D()
    # trefoilW16.plot_density()

    # runs = timeit.repeat(func, number=1000)
