from my_modules.knots_optimization_functions import *


def trefoil_optimization():
    # def test_visual():

    coeffMod = [1.51, -5.06, 7.23, -2.03, -3.97]
    # ['1.53', '-5.04', '7.19', '-1.97', '-3.99']
    coeffStand = [1.715, -5.662, 6.381, -2.305, -4.356]
    coeffTest = [1.56, -5.11, 8.29, -2.37, -5.36]  # this is the best, I think MINE PAPER
    coeffTest = [1.39, -4.42, 7.74, -3.06, -4.08]  # dima 12 dots
    coeffTest = [1.371, -4.1911, 7.9556, -3.4812, -4.2231]  # 12 dots best [1.48, -4.78, 7.15, -2.25, -3.22]
    # coeffTest = [1.828, -5.977, 6.577, -2.347, -3.488]  # w=1.3
    # coeffTest = [1.6321388906295984, -5.0765058505285685, 6.910946967126036, -2.3534680274724753, -4.543804585964645]
    # coeffTest = [1.35, -4.9, 7.43, -2.49, -3.1]
    # coeffTest = [1.41, -3.71, 7.44, -2.09, -4.26]  # dima 6 LAST
    # coeffTest = [1.26, -3.74, 7.71, -2.07, -4.25] # dima 5 dots BEST
    # посмотреть новые для 6 теста [1.41, -3.85, 7.28, -1.95, -4.25]
    # coeffTest /= np.sqrt(sum([a ** 2 for a in coeffTest])) * 0.1
    # coeffTest = np.array(coeffTest) * 1.51 / 1.5308532
    # Mod/Stand=0.95284, Mod/Test=1.03021 (i0=0.05)[1.41, -3.71, 7.44, -2.09, -4.26]
    coeffTest = None
    i0 = 0.01
    iMin = i0 / 100
    xyMinMax = 4
    zMinMax = 1.1  # 2.6
    zRes = 71
    xRes = yRes = 121
    width = 1.3
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)

    # perfect from the paper
    # check_knot_paper(xyzMesh, coeffMod, deltaCoeff=[0.3] * 5, iMin=iMin, i0=i0, radiustest=0.05, steps=1000)
    plot_test = True
    if plot_test:
        xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
        fieldTest = fOAM.trefoil_mod(
            *xyzMesh, w=1.125, width=1.3, k0=1, z0=0.,
            aCoeff=coeffTest, coeffPrint=False
        )
        # fg.plot_3D_density(np.angle(fieldTest))
        fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 2], axis_equal=True)
        fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2], axis_equal=True)
        # fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
        plt.show()
        if 0:
            field, dotsOnly = fg.cut_non_oam(np.angle(fieldTest),
                                             bigSingularity=False, axesAll=False, cbrt=False)
            for i in range(zRes // 2, zRes):
                fg.plot_2D(field[:, :, i])
                plt.show()

        exit()
    check_knot_mine(xyzMesh, coeffTest, deltaCoeff=[0.3] * 5, steps=1,
                    six_dots=True, testvisual=False, width=width,
                    circletest=True, radiustest=0.02,  # # # # # # # # # ## #
                    checkboundaries=True, boundaryValue=0.1,
                    xyzMeshPlot=fg.create_mesh_XYZ(xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
                                                   71, 71, 81, zMin=None))


def hopf_optimization():
    # def test_visual():
    coeffPaper = [2.63, -6.32, 4.21, 5.95]  # у них 5.95
    coeff15 = [2.81, -6.68, 4.29, 5.36]  # w = 1.5 ?
    # coeffTest12 = [3.1183383245351384, -6.487464929941611, 4.539015096229947, 5.339462907691034]  # w=1.2
    # coeffTest13 = [3.205855865528611, -6.434314589371021, 4.627563891019791, 5.325638957032132]  # w=1.3
    # coeffTest14 = [2.995001378425516, -6.519053147580572, 4.462673212141267, 5.378283099690847]  # w=1.4
    coeffTest12 = [2.69, -6.41, 4.48, 5.38]  # w=1.2
    coeffTest125 = [2.73, -6.32, 4.58, 5.34]
    coeffTest13 = [3.09, -6.14, 4.88, 5.54]  # w=1.3
    coeffTest14 = [3.59, -6.31, 5.47, 5.0]  # w=1.4
    # coeffTest14_2 = [ 3.20102509, -6.09079389,  4.98992959,  5.26842205]
    coeff = list(map(lambda x, y: (x + y * 2) / 3, coeffTest12, coeffTest13))
    coeffTest121313 = [2.96, -6.23, 4.75, 5.49]
    coeffTest121313 = [2.96, -6.23, 4.75, 5.49, 0]
    print(coeff)
    coeff = [3.171231805813835, -5.982511722377574, 4.805000010039758, 5.23178437509335, 0.19882157934428407]
    coeff = [3.17, -5.98, 4.81, 5.23, 0.2]
    coeff = [3.19, -6.3, 5.09, 5.04, 0.6]
    coeff = coeffTest121313
    coeff = None
    width = 1.3
    # [3.1183383245351384, -6.487464929941611, 4.539015096229947, 5.339462907691034]
    # [3.212293593009031, -6.392995869495429, 4.629882224195264, 5.242451210074536]
    xyMinMax = 4  # * 2
    zMinMax = 1.1  # * 4
    zRes = 71
    xRes = yRes = 121
    plot_test = True
    if plot_test:
        xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
        fieldTest = fOAM.hopf_mod(
            *xyzMesh, w=1.475, width=width, k0=1, z0=0.,
            coeff=coeff, coeffPrint=True
        )
        # fg.plot_3D_density(np.angle(fieldTest))
        fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 2] ** 2, axis_equal=True)
        fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2], axis_equal=True)
        # fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
        plt.show()
        if 0:
            field, dotsOnly = fg.cut_non_oam(np.angle(fieldTest),
                                             bigSingularity=False, axesAll=False, cbrt=False)
            for i in range(zRes // 2, zRes):
                fg.plot_2D(field[:, :, i])
                plt.show()

        exit()
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)
    check_knot_mine_hopf(xyzMesh, coeff, deltaCoeff=[0.2] * 5, steps=1,
                         six_dots=False, testvisual=False, width=width,
                         circletest=False, radiustest=0.02,  # # # # # # # # # ## #
                         checkboundaries=True, boundaryValue=0.1,
                         xyzMeshPlot=fg.create_mesh_XYZ(xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2,
                                                        51, 51, 141, zMin=None))


trefoil_optimization()
# hopf_optimization()
