import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_high_lvl as fhl
import functions_OAM_knots as fOAM
import knot_class as kc

if __name__ == '__main__':
    knot_optimization = 0
    if knot_optimization:
        fhl.knot_optimization()

    # not finished
    propagation_modification = True
    if propagation_modification:
        def propagation_modification():
            xyMinMax = 5
            xRes = yRes = 30
            xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
            yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
            # xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0.5, xRes, yRes, 7)
            xyMesh = np.meshgrid(xArray, yArray)
            field = fOAM.knot_all(*xyMesh, 0, w=1.2, width=1.2, k0=1, z0=0., knot=None)
            # fg.plot_2D(np.abs(ifftshift(fftn(field[:, :, np.shape(field)[2] // 2]))),
            #           xlim=[20, 30], ylim=[20, 30])
            # fg.plot_2D(np.abs(field))
            # fg.plot_2D(np.angle(field))
            # fg.plot_2D(np.real(field))
            # fg.plot_2D(np.imag(field))
            multK = 3.5
            kxArray = np.linspace(-2 * multK * np.pi / xyMinMax, 2 * multK * np.pi / xyMinMax, xRes)
            kyArray = np.linspace(-2 * multK * np.pi / xyMinMax, 2 * multK * np.pi / xyMinMax, yRes)
            fieldSpec = fg.ft_forward_2D(field, xArray, yArray, kxArray, kyArray)
            # fg.plot_2D(np.abs(fieldSpec))
            # fg.plot_2D(np.abs(np.fft.fftn(np.fft.fftn(field[:, :, np.shape(field)[2] // 2]))))
            multXY = 2
            xArrayNew = np.linspace(-xyMinMax * multXY, xyMinMax * multXY, xRes * 1)
            yArrayNew = np.linspace(-xyMinMax * multXY, xyMinMax * multXY, yRes * 1)
            fieldHiger = fg.ft_reverse_2D(fieldSpec, xArrayNew, yArrayNew, kxArray, kyArray)
            fg.plot_2D(np.abs(fieldHiger))
            fg.plot_2D(np.angle(fieldHiger))
            fg.plot_2D(np.angle(fg.cut_fourier_filter(fieldHiger, 20)))
            # fg.plot_2D(np.angle(np.fft.fftn(fieldSpec)))

            # fg.plot_2D(np.abs(np.fft.fftshift(np.fft.fftn(field))))
            # fg.plot_2D(np.abs(np.fft.ifftn(fieldSpec)))
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


        propagation_modification()

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

    creating_table_knots = 1  # making_table1

    if creating_table_knots:
        SR = '0.95'
        knot = 'Trefoil'
        w = '1.2 6dots no 12'  # Dima Cmex-
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
