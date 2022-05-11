import matplotlib.pyplot as plt
import numpy as np
import os
import sympy
import pyknotid.spacecurves as sp
import pandas as pd

import functions_general as fg


def vectorDotProduct(vector1, vector2):
    return vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]


def vectorAbs(vector):
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def distancePoints2D(point1, point2):
    deltaX = point1[0] - point2[0]
    deltaY = point1[1] - point2[1]
    return np.sqrt(deltaX ** 2 + deltaY ** 2)


def distancePoints(point1, point2):
    deltaX = point1[0] - point2[0]
    deltaY = point1[1] - point2[1]
    deltaZ = point1[2] - point2[2]
    return np.sqrt(deltaX ** 2 + deltaY ** 2 + deltaZ ** 2)


def vectorAngle2D(vector1, vector2):
    vector1[2] = 0
    vector2[2] = 0
    return np.arccos(vectorDotProduct(vector1, vector2) / (vectorAbs(vector1) * vectorAbs(vector2)))


def maxDistance(dots, mid=None, twoD=True):
    if mid is None:
        mid = [0, 0, 0]
    answer = 0
    for i in range(len(dots)):
        if twoD:
            currentDistance = distancePoints2D(dots[i][0:2], mid[0:2])
        else:
            currentDistance = distancePoints(dots[i], mid)
        if currentDistance > answer:
            answer = currentDistance
    return answer


def cleanLayers(weirdArrayWithZeros, zArray=None, cleanLayers=True):
    if zArray is None:
        zArray = np.arange(len(weirdArrayWithZeros))
    cleanedLayers = []
    for i in range(len(weirdArrayWithZeros)):
        if cleanLayers:
            if weirdArrayWithZeros[i].size != 0:
                newZPlaneArray = np.zeros((len(weirdArrayWithZeros[i]), 3))
                for j in range(len(weirdArrayWithZeros[i])):
                    newZPlaneArray[j] = np.append(weirdArrayWithZeros[i][j], zArray[i])
                cleanedLayers.append(newZPlaneArray)
        else:
            newZPlaneArray = np.zeros((len(weirdArrayWithZeros[i]), 3))
            for j in range(len(weirdArrayWithZeros[i])):
                newZPlaneArray[j] = np.append(weirdArrayWithZeros[i][j], zArray[i])
            cleanedLayers.append(newZPlaneArray)
    cleanedLayers.extend([[]])
    if cleanLayers:
        return np.array(cleanedLayers, dtype=object)
    else:
        return np.delete(np.array(cleanedLayers, dtype=object), 0)


def makeDots(dotsArray, zArray):
    dotsWithZ = np.zeros((1, 3))
    zIndex = 0
    for dotsZ in dotsArray:
        zValue = zArray[zIndex]
        zIndex += 1
        for dotZ in dotsZ:
            if len(dotZ) != 0:
                dotsWithZ = np.append(dotsWithZ, np.array([np.append(dotZ, zValue)]), axis=0)
    return np.delete(dotsWithZ, 0, 0)


def deleteGarbageDots(dots, distanceNorm=1, mid=None, twoD=True):
    answer = dots
    if mid is None:
        mid = [0, 0, 0]
    delDist = maxDistance(dots, mid=mid, twoD=twoD) * distanceNorm
    removed = 0
    for i in range(len(dots)):
        if twoD:
            currentDistance = distancePoints2D(dots[i][0:2], mid[0:2])
        else:
            currentDistance = distancePoints(dots[i], mid)
        if currentDistance > delDist:
            answer = np.delete(answer, i - removed, axis=0)
            removed += 1
    return answer


def printDots(dots, dotsNumber=None):
    if dotsNumber is None:
        dotsNumber = len(dots)
    for i in range(dotsNumber):
        print(f"(dot #{i}) z={dots[i][2]}:\t(x={dots[i][0]}, y={dots[i][1]})", )


def plotDots(dots):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2])  # plot the point (2,3,4) on the figure
    ax.view_init(70, 0)
    # plt.show()
    # plt.close()


class Knot3(object):
    def __init__(self, dotsArray=None, dotsFileName=None, dz=1, clean=1, angleCheck=180, distCheck=10, layersStep=1):
        self.dz = dz
        self.angleCheck = angleCheck
        if dotsArray is None:
            dotsArray = np.array(fg.readingFile(dotsFileName)[0, :])
            print(f"number of layers: {len(dotsArray)}")
            self.zArray = np.array(range(len(dotsArray))) * self.dz
            self.zArray = self.zArray - self.zArray.max() / 2
            self.initialDots = makeDots(dotsArray, self.zArray)
            self.layers = cleanLayers(dotsArray, self.zArray, cleanLayers=False)
        else:
            self.zArray = dotsArray[:, 2] * dz
            self.initialDots = dotsArray
        self.dots = self.initialDots
        # self.dotsDeleted = np.array([[1, 1, 1]])
        # self.actualKnotLength = len(self.dots) - 1
        self.distCheck = distCheck
        self.layersStep = layersStep
        self.knot = []
        # self.dotsDeleted = np.array([[1, 1, 1]])
        self.initialProcedures(clean=clean)
        self.planes = []

    def initialProcedures(self, clean=1):
        # sorting the dots in z
        self.dots = np.array([self.dots[i] for i in np.argsort(self.dots[:, 2], axis=0)])
        # self.dotsDeleted = np.delete(self.dotsDeleted, 0, axis=0)
        # don't know how to make the right array size so adding and deleting te element
        # self.dotsDeleted = np.delete(self.dotsDeleted, 0, axis=0)
        if clean != 1:  # Removing a cylinder of garbage dots
            self.deleteGarbageDots(distanceNorm=clean)
        if len(self.dots) <= 2:
            self.dots = self.initialDots
        self.knot = np.array([self.dots[0]])
        # first dot is in the knot
        self.dots = np.delete(self.dots, 0, axis=0)
        # self.dotsDeleted = np.append(self.dotsDeleted, [self.knot[-1]], axis=0)

    def creatingKnot2(self, knotLength=None):
        if knotLength is None:
            knotLength = len(self.dots)
        while len(self.knot) < knotLength:
            if not len(self.dots):
                break
            if not self.addClosest():
                if len(self.knot) <= 2:
                    self.knot = np.append(self.knot, [self.dots[-1]], axis=0)
                    self.knot = np.append(self.knot, [self.dots[-2]], axis=0)
                    # self.dots = np.delete(self.dots, tempIndex, axis=0)
                break
        # print(f"removed dots amount: {len(self.dotsDeleted)}")

    def creatingKnotFromGarbage(self, knotLength=None):
        self.knot = np.array([self.dots[0]])
        self.dots = np.delete(self.dots, 0, axis=0)
        if knotLength is None:
            knotLength = len(self.dots)
        while len(self.knot) < knotLength:
            if not len(self.dots):
                break
            if not self.addClosest():
                if len(self.knot) <= 2:
                    self.knot = np.append(self.knot, [self.dots[-1]], axis=0)
                    self.knot = np.append(self.knot, [self.dots[-2]], axis=0)
                    # self.dots = np.delete(self.dots, tempIndex, axis=0)
                break

    def addClosest(self):
        tempIndex = -19
        tempDistance = self.distCheck * self.dz
        foundDot = False
        for i in range(len(self.dots)):
            if abs(self.dots[i][2] - self.knot[-1][2]) < (
                    self.layersStep + 1) * self.dz:  # and self.dots[i][2] - self.knot[-1][2] != 0:  #self.dots[i][2] != self.knot[-1][2] or self.dots[i][2] == self.knot[-1][2]
                distance = distancePoints2D(self.dots[i], self.knot[-1])  # distancePoints2D
                if distance < tempDistance:
                    tempDistance = distance
                    tempIndex = i
                    foundDot = True
        if foundDot:
            self.knot = np.append(self.knot, [self.dots[tempIndex]], axis=0)
            self.dots = np.delete(self.dots, tempIndex, axis=0)
            self.knotCurrentAngleCheck()
            return True
        else:
            print(f"\033[03m We haven't found a dot:\n\t\t current length: {len(self.knot)}, "
                  f"total amount: {len(self.initialDots)} (before cleaning)")
            return False

    # removing the dots in a circle
    def deleteGarbageDots(self, distanceNorm=1, mid=None, twoD=True):
        self.dots = deleteGarbageDots(self.dots, distanceNorm=distanceNorm, mid=mid, twoD=twoD)

    def printDots(self, dotsNumber=None):
        if dotsNumber is None:
            dotsNumber = len(self.dots)
        printDots(self.dots, dotsNumber)

    def printKnot(self, dotsNumber=None):
        if dotsNumber is None:
            dotsNumber = len(self.knot)
        printDots(self.knot, dotsNumber)

    def knotCurrentAngleCheck(self):
        if len(self.knot) > 3:
            vector1 = self.knot[-2] - self.knot[-3]
            vector2 = self.knot[-1] - self.knot[-2]
            if np.degrees(vectorAngle2D(vector1, vector2)) > self.angleCheck:
                # print(f"angle : {np.degrees(vectorAngle2D(vector1, vector2))}")
                # self.dotsDeleted = np.append(self.dotsDeleted, [self.knot[-1]], axis=0)
                self.knot = np.delete(self.knot, -1, axis=0)

    def makePlanes(self, around=False, fileName=None):
        maxDist = maxDistance(self.initialDots, twoD=True)
        print(maxDist)
        newPlane = [self.initialDots[0]]
        for dot in self.initialDots[1:-1]:
            if dot[2] == newPlane[0][2]:
                newPlane.append(dot)
            else:
                self.planes.append(newPlane)
                newPlane = [dot]
        index = -1
        newPlanes = []
        for plane in self.planes[::1]:
            index += 1
            print(f'index: {index}')
            dotInd = 0
            for dot in plane:
                plt.scatter(dot[0], dot[1], s=300, c='r')
                plt.text(dot[0], dot[1], f'{dotInd}', fontsize=25)
                dotInd += 1
            if around:
                if len(newPlanes) != 0:  # index != 0:
                    for dot in newPlanes[-1]:  # self.planes[index - 1]:
                        plt.scatter(dot[0], dot[1], s=120, c='g')
                if index != len(self.planes) - 1:
                    for dot in self.planes[index + 1]:
                        plt.scatter(dot[0], dot[1], s=150, c='b', marker='x')
                    if index != len(self.planes) - 2:
                        for dot in self.planes[index + 2]:
                            plt.scatter(dot[0], dot[1], s=150, c='y', marker='+')
            plt.title(f'z={plane[0][2]}, g=prv, b=next')
            plt.xlim(0, maxDist)
            plt.ylim(0, maxDist)
            plt.show()
            plt.close()
            inp = int(input())
            if inp == 1:
                newPlanes.append(plane)
        newDots = np.zeros((1, 3))
        for plane in newPlanes:
            for dot in plane:
                newDots = np.append(newDots, [dot], axis=0)
        newDots = np.delete(newDots, 0, axis=0)
        self.dots = newDots
        if fileName is not None:
            np.save(fileName, self.dots)
        self.knot = np.array([self.dots[0]])
        self.dots = np.delete(self.dots, 0, axis=0)

        # plt.plot()

    def plotKnotDots(self):
        # fig = plt.figure(figsize=(4, 4))
        plotDots(self.knot)

    def plotDots(self, initial=0):
        # fig = plt.figure(figsize=(4, 4))
        if initial:
            plotDots(self.initialDots)
        else:
            plotDots(self.dots)


class Knots(object):
    def __init__(self, knot=None, alexType=None):
        self.knots = []
        if knot is not None:
            self.knots.append(knot.knot)
        if alexType is None:
            t = sympy.symbols("t")
            k = sp.Knot(knot.knot, add_closure=False)
            self.alex = k.alexander_polynomial(variable=t)
        else:
            self.alex = alexType

    def addKnot(self, knot):
        if len(self.knots[0]) / 3 < len(knot.knot):
            self.knots.append(knot.knot)


def making_knot(name, show=True, cut=1):
    knot = Knot3(dotsFileName=name, dz=4, clean=cut, angleCheck=180, distCheck=5, layersStep=1)
    knot.plotDots(initial=0)
    try:
        knot.creatingKnot2(knotLength=None)
        knots = Knots(knot)
        k = sp.Knot(knots.knots[0] / [[1, 1, 1]], add_closure=True)
        # k.plot(mode='matplotlib')

        # knots = Knots(knot)
        # print(knots.alex)
        t = sympy.symbols("t")

        ap = k.alexander_polynomial(variable=t)
        print(ap)
        if ap != -t ** 2 + t - 1 and ap != -t ** 4 + t ** 3 - t ** 2 + t - 1 and ap != t ** 6 - t ** 5 + t ** 4 - t ** 3 + t ** 2 - t + 1:
            print("Weird")
            # plt.savefig(directoryNamePlots + fileName[:-4] + '.png', format="png")

            if show:
                plt.show()
        else:
            plt.close()
        return ap
    except Exception:
        print('we are bad')
        #plt.savefig(directoryNamePlots + fileName[:-4] + '.png', format="png")
        if show:
            plt.show()
        return 0


def creat_knot_table(directoryName, tableName, show=True, cut=1):
    listOfFiles = [f for f in os.listdir(directoryName) if f.endswith(".mat")]
    list_of_file_names = []
    list_of_file_names_bad = []
    list_of_alex = []
    for fileName in listOfFiles[::]:
        print(fileName[:-4])
        pathName = directoryName[-20:-0]
        ap = making_knot(directoryName + fileName, show=show, cut=cut)

        t = sympy.symbols("t")
        if ap == t ** 6 - t ** 5 + t ** 4 - t ** 3 + t ** 2 - t + 1:
            list_of_file_names.append(f'{pathName}{fileName}')
            list_of_alex.append(str(ap))
        elif ap == -t ** 4 + t ** 3 - t ** 2 + t - 1:  # -t**2 + t - 1
            list_of_file_names.append(f'{pathName}{fileName}')
            list_of_alex.append(str(ap))
        elif ap == -t ** 2 + t - 1:
            list_of_file_names.append(f'{pathName}{fileName}')
            list_of_alex.append(str(ap))
            # list_of_alex.append(str(ap))9
        elif show:
            list_of_file_names.append(f'{pathName}{fileName}')
            formula = input()

            list_of_alex.append(formula)
        else:
            list_of_file_names.append(f'{pathName}{fileName}')
            list_of_alex.append(str(ap))
        plt.close('all')

    good = pd.DataFrame({'file': list_of_file_names,
                         'alex': list_of_alex})
    bad = pd.DataFrame({'file': list_of_file_names_bad})
    knot_sheets = {'good': good, 'bad': bad}
    writer = pd.ExcelWriter(f'{tableName}.xlsx', engine='xlsxwriter')

    for sheet_name in knot_sheets.keys():
        knot_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()
