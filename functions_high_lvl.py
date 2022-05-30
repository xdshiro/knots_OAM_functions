import functions_general as fg
import numpy as np
import matplotlib.pyplot as plt
import functions_OAM_knots as fOAM
from knots_optimization import *

# used for the comparison between different sizes of the phase screens for knots
def knot_from_math_f():
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
    # fOAM.plot_knot_dots(field1)
    plt.show()


def propagation_modification():
    xyMinMax = 5
    xRes = yRes = 50
    xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
    yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0.5, xRes, yRes, 7)
    field = fOAM.knot_all(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)
    # fg.plot_2D(np.abs(ifftshift(fftn(field[:, :, np.shape(field)[2] // 2]))),
    #           xlim=[20, 30], ylim=[20, 30])
    kxArray = np.linspace(-4 * 2 * np.pi / xyMinMax, 4 * 2 * np.pi / xyMinMax, xRes)
    kyArray = np.linspace(-4 * 2 * np.pi / xyMinMax, 4 * 2 * np.pi / xyMinMax, yRes)
    fieldSpec = fg.ft_2D(field, xArray, yArray, kxArray, kyArray)
    # fg.plot_2D(np.abs(np.fft.fftn(np.fft.fftn(field[:, :, np.shape(field)[2] // 2]))))
    xArrayNew = np.linspace(-xyMinMax * 2, xyMinMax * 2, xRes * 2)
    yArrayNew = np.linspace(-xyMinMax * 2, xyMinMax * 2, yRes * 2)
    field = fg.ft_2D(fieldSpec, kxArray, kyArray, xArrayNew, yArrayNew)
    fg.plot_2D(np.abs(field))
    # fg.plot_2D(np.abs(np.fft.ifftshift(np.fft.ifftn(fieldSpec))))
    plt.show()
    exit()
    fieldProp = fg.propagator_split_step_3D(field[:, :, np.shape(field)[2] // 2],
                                            xArray=xArray, yArray=yArray,
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
    field_inter_prop = fg.propagator_split_step_3D(field_inter, dz=0.2, zSteps=30)
    fg.plot_2D(np.abs(field_inter_prop[:, :, -1]))
    fg.plot_2D(np.angle(field_inter_prop[:, :, -1]))
    plt.show()


def milnor_research():
    xyzMesh = fg.create_mesh_XYZ(2, 2, 0.5, 40, 40, 40)
    field = fOAM.milnor_Pol_testing(xyzMesh[0], xyzMesh[1], xyzMesh[2], 2, 3)
    fg.plot_2D(np.abs(field[:, :, np.shape(field)[2] // 2]))
    fg.plot_2D(np.angle(field[:, :, np.shape(field)[2] // 2]))
    fOAM.plot_knot_dots(field, color='k', size=80, axesAll=True)
    # field = fOAM.knot_all(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, knot='trefoil_mod')
    # fg.plot_3D_density(np.angle(field))
    # fg.plot_2D(np.angle(field[:, :, np.shape(field)[2] // 2]))
    plt.show()


def knot_optimization():
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
    i0 = 0.01
    iMin = i0 / 100
    xyMinMax = 4
    zMinMax = 1.1  # 2.6
    zRes = 121
    xRes = yRes = 101
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)

    # perfect from the paper
    # check_knot_paper(xyzMesh, coeffMod, deltaCoeff=[0.3] * 5, iMin=iMin, i0=i0, radiustest=0.05, steps=1000)
    plot_test = True
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
        plt.show()
        if 1:
            field, dotsOnly = fg.cut_non_oam(np.angle(fieldTest),
                                             bigSingularity=False, axesAll=False, cbrt=False)
            for i in range(zRes // 2, zRes):
                fg.plot_2D(field[:, :, i])
                plt.show()

        exit()
    check_knot_mine(xyzMesh, coeffTest, deltaCoeff=[0.3] * 5, steps=5000,
                    six_dots=True, testvisual=True,
                    circletest=True, radiustest=0.02,  # # # # # # # # # ## #
                    checkboundaries=True, boundaryValue=0.1,
                    xyzMeshPlot=fg.create_mesh_XYZ(xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
                                                   71, 71, 81, zMin=None))


def hopf_optimization():
    # def test_visual():
    coeffPaper = [2.63, -6.32, 4.21, -5.95]
    coeff15 = [2.81, -6.68, 4.29, 5.36]
    coeffTest = [3.1183383245351384, -6.487464929941611, 4.539015096229947, 5.339462907691034]  # w=1.2
    coeffTest = [3.205855865528611, -6.434314589371021, 4.627563891019791, 5.325638957032132]  # w=1.3
    coeff = coeffTest
    # [3.1183383245351384, -6.487464929941611, 4.539015096229947, 5.339462907691034]
    # [3.212293593009031, -6.392995869495429, 4.629882224195264, 5.242451210074536]
    xyMinMax = 4
    zMinMax = 1.1  # 2.6
    zRes = 81
    xRes = yRes = 151
    plot_test = True
    if plot_test:
        xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)
        fieldTest = fOAM.hopf_mod(
            *xyzMesh, w=1.5, width=1.3, k0=1, z0=0.,
            coeff=coeff, coeffPrint=True
        )
        # fg.plot_3D_density(np.angle(fieldTest))
        fg.plot_2D(np.abs(fieldTest)[:, :, np.shape(fieldTest)[2] // 2] ** 2, axis_equal=True)
        fg.plot_2D(np.angle(fieldTest)[:, :, np.shape(fieldTest)[2] // 2], axis_equal=True)
        fOAM.plot_knot_dots(fieldTest, axesAll=True, color='r', size=200)
        plt.show()
        if 1:
            field, dotsOnly = fg.cut_non_oam(np.angle(fieldTest),
                                             bigSingularity=False, axesAll=False, cbrt=False)
            for i in range(zRes // 2, zRes):
                fg.plot_2D(field[:, :, i])
                plt.show()

        exit()
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=0)
    check_knot_mine_hopf(xyzMesh, coeff, deltaCoeff=[0.05] * 4, steps=5000,
                         six_dots=False, testvisual=False,
                         circletest=False, radiustest=0.02,  # # # # # # # # # ## #
                         checkboundaries=True, boundaryValue=0.1,
                         xyzMeshPlot=fg.create_mesh_XYZ(xyMinMax * 1.3, xyMinMax * 1.3, zMinMax * 2.5,
                                                        71, 71, 81, zMin=None))