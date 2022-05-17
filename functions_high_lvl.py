import functions_general as fg
import numpy as np
import matplotlib.pyplot as plt
import functions_OAM_knots as fOAM


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

