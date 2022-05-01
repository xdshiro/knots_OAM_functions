import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import scipy.io as sio

# common parameters
# global fig, ax
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def resolutionDecrease(E, resDecrease=None):
    if resDecrease is None:
        resDecrease = [1, 1, 1]
    newE = E[::resDecrease[0], ::resDecrease[1], ::resDecrease[2]]
    return newE


def cut_middle_values(E, min, max, midValue=0, minValue=None, maxValue=None):
    ans = np.copy(E)
    minFlag, maxFlag = False, False
    if minValue is not None:
        minFlag = True
    if maxValue is not None:
        maxFlag = True
    shape = np.shape(ans)
    if len(shape) == 1:
        for i in range(shape[0]):
            if ans[i] <= min:
                if minFlag:
                    ans[i] = minValue
            elif ans[i] >= max:
                if maxFlag:
                    ans[i] = maxValue
            else:
                ans[i] = midValue
    else:
        for i in range(shape[0]):
            ans[i] = cut_middle_values(ans[i], min, max, midValue, minValue=minValue, maxValue=maxValue)
    return ans


def check_dot_oam(E):
    flagPlus, flagMinus = True, True
    minIndex = np.argmin(E)
    for i in range(minIndex - len(E), minIndex - 1, 1):
        if E[i] >= E[i + 1]:
            flagMinus = False
            break
    maxIndex = np.argmax(E)
    for i in range(maxIndex - len(E), maxIndex - 1, 1):
        if E[i] <= E[i + 1]:
            flagPlus = False
            break
    if flagPlus:
        return True, +1
    elif flagMinus:
        return True, -1
    return False, 0


def fill_dict_as_matrix(E, dots=None, nonValue=0, check=False):
    if dots is None:
        dots = {}
    shape = np.shape(E)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if E[i, j, k] != nonValue:
                    if check:
                        if dots.get((i, j, k)) is None:
                            dots[(i, j, k)] = E[i, j, k]
                    else:
                        dots[(i, j, k)] = E[i, j, k]
    return dots


def cut_non_oam(E, value=1, nonValue=0, bigSingularity=False, axesAll=False, cbrt=False):
    ans = np.copy(E)
    shape = np.shape(ans)
    dots = {}
    if len(shape) == 2:
        for i in range(1, shape[0] - 1, 1):
            for j in range(1, shape[1] - 1, 1):
                Echeck = np.array([E[i - 1, j - 1], E[i - 1, j], E[i - 1, j + 1],
                                   E[i, j + 1], E[i + 1, j + 1], E[i + 1, j],
                                   E[i + 1, j - 1], E[i, j - 1]])
                # if len(Echeck[Echeck>2.0]) != 0 and len(Echeck[Echeck<-2.0]) != 0:
                #    print(Echeck)
                oamFlag, oamValue = check_dot_oam(Echeck)
                # print(oamFlag, oamValue)
                if oamFlag:
                    ans[i, j] = oamValue * value
                    if bigSingularity:
                        ans[i - 1:i + 2, j - 1:j + 2] = oamValue * value
                else:
                    ans[i, j] = nonValue

        ans[:1, :] = nonValue
        ans[-1:, :] = nonValue
        ans[:, :1] = nonValue
        ans[:, -1:] = nonValue
    else:
        for i in range(shape[2]):
            ans[:, :, i] = cut_non_oam(ans[:, :, i], value=value, nonValue=nonValue,
                                       bigSingularity=bigSingularity)[0]
        dots = fill_dict_as_matrix(ans)

        if axesAll:
            for i in range(shape[1]):
                ans[:, i, :] += cut_non_oam(E[:, i, :], value=value, nonValue=nonValue,
                                            bigSingularity=bigSingularity)[0]
            dots = fill_dict_as_matrix(ans, dots, check=True)
            for i in range(shape[0]):
                ans[i, :, :] += cut_non_oam(E[i, :, :], value=value, nonValue=nonValue,
                                            bigSingularity=bigSingularity)[0]
            dots = fill_dict_as_matrix(ans, dots, check=True)
            if cbrt:
                ans = np.array(np.cbrt(ans), dtype=int)

    # print(ans)
    return [ans, dots]


def cut_fourier_filter(E, radiusPix=1):
    ans = np.copy(E)
    ans = fftshift(fftn(ans))
    xCenter, yCenter = np.shape(ans)[0] // 2, np.shape(ans)[0] // 2
    for i in range(np.shape(ans)[0]):
        for j in range(np.shape(ans)[1]):
            if np.sqrt((xCenter - i) ** 2 + (yCenter - j) ** 2) > radiusPix:
                ans[i, j] = 0
    ans = ifftn(ifftshift(ans))
    print(np.shape(ans))
    return ans


def simple_propagator_3D(E, dz=1, xArray=None, yArray=None, zSteps=1, n0=1, k0=1):
    if xArray is None:
        xArray = range(np.shape(E)[0])
    if yArray is None:
        yArray = range(np.shape(E)[1])
    xResolution, yResolution = len(xArray), len(yArray)
    zResolution = zSteps + 1
    intervalX = xArray[-1] - xArray[0]
    intervalY = yArray[-1] - yArray[0]

    # xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
    kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / intervalX,
                          1. * np.pi * (xResolution - 2) / intervalX, xResolution)
    kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / intervalY,
                          1. * np.pi * (yResolution - 2) / intervalY, yResolution)

    KxyMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

    def Nonlinearity_spec(E):
        return dz * 0

    # works fine!
    def linear_step(field):
        temporaryField = np.fft.fftshift(np.fft.fftn(field))
        temporaryField = (temporaryField *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[0] ** 2) *
                          np.exp(-1j * dz / (2 * k0 * n0) * KxyMesh[1] ** 2))  # something here in /2
        return np.fft.ifftn(np.fft.ifftshift(temporaryField))

    fieldReturn = np.zeros((xResolution, yResolution, zResolution), dtype=complex)
    fieldReturn[:, :, 0] = E
    for k in range(1, zResolution):
        fieldReturn[:, :, k] = linear_step(fieldReturn[:, :, k - 1])
        fieldReturn[:, :, k] = fieldReturn[:, :, k] * np.exp(Nonlinearity_spec(fieldReturn[:, :, k]))

    return fieldReturn


def cut_fourier_filter_3D(E, radiusPix=1):
    ans = np.copy(E)
    ans = fftshift(fftn(ans))
    xCenter, yCenter, zCenter = np.shape(ans)[0] // 2, np.shape(ans)[1] // 2, np.shape(ans)[2] // 2
    for i in range(np.shape(ans)[0]):
        for j in range(np.shape(ans)[1]):
            for k in range(np.shape(ans)[2]):
                if np.sqrt((xCenter - i) ** 2 + (yCenter - j) ** 2 + 3 * (zCenter - k) ** 2) > radiusPix:
                    ans[i, j, k] = 0
    ans = ifftn(ifftshift(ans))
    return ans


def readingFile(fileName, fieldToRead="p_charges", printV=False):
    matFile = sio.loadmat(fileName, appendmat=False)
    if printV:
        print(matFile)
        exit()
    return np.array(matFile[fieldToRead])


def plot_3D_density(E, resDecrease=None,
                    xMinMax=None, yMinMax=None, zMinMax=None,
                    surface_count=2,
                    opacity=1,
                    opacityscale=None):
    if zMinMax is None:
        zMinMax = [0, 1]
    if yMinMax is None:
        yMinMax = [0, 1]
    if xMinMax is None:
        xMinMax = [0, 1]
    values = resolutionDecrease(E, resDecrease)
    shape = np.array(np.shape(E))
    if resDecrease is not None:
        shape = (shape // resDecrease)
    X, Y, Z = np.mgrid[xMinMax[0]:xMinMax[1]:shape[0] * 1j,
              yMinMax[0]:yMinMax[1]:shape[1] * 1j,
              zMinMax[0]:zMinMax[1]:shape[2] * 1j]
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),  # collapsed into 1 dimension
        y=Y.flatten(),
        z=Z.flatten(),
        value=values.flatten(),
        isomin=values.min(),
        isomax=values.max(),
        opacity=opacity,  # needs to be small to see through all surfaces
        opacityscale=opacityscale,
        surface_count=surface_count,  # needs to be a large number for good volume rendering
        colorscale='RdBu'
    ))
    fig.show()


def plotDots(dots):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2])  # plot the point (2,3,4) on the figure
    ax.view_init(70, 0)
    # plt.show()
    # plt.close()


def plot_1D(x, y, noX=False, label=None, xname='', yname='', ls='-', lw=4, color=None,
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            legendFontSize=legendFontSize,
            marker='', markersize=12, markeredgecolor='k',
            loc='lower right',
            axis_equal=False,
            ax=None):
    if noX:
        x = range(len(y))
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x, y, linestyle=ls,
            marker=marker, markersize=markersize, markeredgecolor=markeredgecolor,
            label=label, lw=lw, color=color)
    """    if dotsColor == 'rrr':
        plt.scatter(x, y, s=dotsSize, label=False)
    else:
        plt.scatter(x, y, s=dotsSize, label=False, color=color)"""
    plt.xticks(fontname='Times New Roman', fontsize=ticksFontSize)
    plt.yticks(fontname='Times New Roman', fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    ax.set_ylabel(yname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if label:
        legend = ax.legend(shadow=True, fontsize=legendFontSize,
                           facecolor='white', edgecolor='black', loc=loc)
        plt.setp(legend.texts, family='Times New Roman')


def plot_2D(E, x=None, y=None, xname='', yname='', map='jet', vmin=None, vmax=None, title='',
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            axis_equal=False,
            xlim=None, ylim=None, ax=None):
    if x is None:
        x = range(np.shape(E)[0])
    if y is None:
        y = range(np.shape(E)[1])
    if ax is None:
        fig, ax = plt.subplots()
    image = plt.imshow(E,
                       interpolation='bilinear', cmap=map,
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[y[0], y[-1], x[0], x[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd')
    cbr = plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if axis_equal:
        ax.set_aspect('equal', adjustable='box')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.tight_layout()
    return ax


def plot_scatter_3D(X, Y, Z, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)  # plot the point (2,3,4) on the figure
    # ax.view_init(70, 0)
    # plt.show()
    # plt.close()
    return ax
