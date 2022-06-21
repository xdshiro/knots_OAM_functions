import numpy as np
import matplotlib.pyplot as plt

# mine
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM
import os
#
# xyMinMax = 5
#
# xRes = yRes = 70
# xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
# yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
# xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0, xRes, yRes, 1)
# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=1) \
#         + 2.5 * fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=0)
# fg.plot_2D(np.angle(field))
# fg.plot_2D(np.abs(field))
# fg.Jz_calc(field)
# plt.show()



#
# intensity = (fg.readingFile(f'.\\PhaseRetrieval 1\\intensity.mat', fieldToRead='Intensity'))
# fg.plot_2D(intensity)
# plt.show()
# intensity = np.sqrt(intensity)
# tem = np.copy(intensity[0:50, 0:400])
# print(tem)
# print(tem.max(), tem.min())
# exit()
# for i in range(8):
#     intensity[(50 * i):(50 * i + 50), 0:400] -= tem
# fg.plot_2D(np.abs(intensity))
# phase = fg.readingFile(f'.\\PhaseRetrieval 1\\phase.mat', fieldToRead='phase')
# fg.plot_2D(phase)
#
# # plt.show()
# field = intensity
#
# # intensityP = intensity[intensity > 0]
# # intensityN = intensity[intensity <= 0]
# # fieldAbsP = np.sqrt(intensityP)
# # fieldAbsN = - np.sqrt(- intensityN)
# # fieldAbs = fieldAbsP + fieldAbsN
# field = field * np.exp(1j * phase)
# fg.plot_2D(np.abs(field))
# plt.show()
# fg.Jz_calc(field)


M1 = np.array([[1, 2], [3, 4]])
M2 = np.array([[2, 4], [6, 8]])
print(M1 * M2)
exit()
intensity = np.double(fg.readingFile(f'.\\PhaseRetrieval 1\\intensity.mat', fieldToRead='Intensity'))

tem = np.copy(intensity[0:50, 0:400])
for i in range(8):
    intensity[(50 * i):(50 * i + 50), 0:400] -= tem
fg.plot_2D(intensity)
# fg.plot_2D(np.abs(intensity))
phase = fg.readingFile(f'.\\PhaseRetrieval 1\\phase.mat', fieldToRead='phase')
fg.plot_2D(phase)

# plt.show()
shape = np.shape(intensity)
field = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        if intensity[i, j] < 0:
            field[i, j] = - np.sqrt(-intensity[i, j])
        # elif intensity[i, j] == 0:
        #     field[i, j] = 1
        else:
            field[i, j] = np.sqrt(intensity[i, j])


# intensityP = intensity[intensity > 0]
# intensityN = intensity[intensity <= 0]
# fieldAbsP = np.sqrt(intensityP)
# fieldAbsN = - np.sqrt(- intensityN)
# fieldAbs = fieldAbsP + fieldAbsN
field = field * np.exp(1j * phase)
fg.plot_2D(np.abs(field))
plt.show()
fg.Jz_calc(field)
fg.Jz_calc_no_conj(field)

# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)

# xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
# yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
# xArray = list(range(np.shape(field)[0]))
# yArray = list(range(np.shape(field)[1]))
# fg.Jz_calc(field, xArray, yArray)
# fg.plot_2D(np.abs(field) ** 2, axis_equal=True)
# plt.show()
