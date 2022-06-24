import numpy as np
import matplotlib.pyplot as plt

# mine
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM
import os

xyMinMax = 5

xRes = yRes = 120
xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0, xRes, yRes, 1)
# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=2) \
#         + fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=0, p=2)
field = fOAM.trefoil_mod(
            *xyzMesh, w=1.6, width=1.3, k0=1, z0=0.,
            aCoeff=[2.015, -6.403, 6.770, -2.359, -1.881], coeffPrint=True
        )
fg.plot_2D(np.angle(field))
fg.plot_2D(np.abs(field))
fg.Jz_calc_no_conj(field)
fg.Jz_calc(field)
plt.show()

exit()
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
# U = padarray(U,[pad,pad],'both');
# Это увеличивает разрешение поля
directory = './/PhaseRetrieval 1'
field = fg.readingFile(directory + f'\\field.mat', fieldToRead='U')
i1, i2 = 712 - 165, 712 + 165
field = field[i1:i2, i1:i2]
fg.plot_2D(np.abs(field))
fg.plot_2D(np.angle(field))
plt.show()
fg.Jz_calc(field)
Jz = fg.Jz_calc_no_conj(field)
W = fg.W_energy(field)
OAM = Jz / W
dEr, dEi = np.real(field) * 0.1, np.imag(field) * 0.1
# dEr = dEi = np.ones(np.shape(field))
# print(abs(field[:, 65]))
# exit()
dW = fg.deltaW(field, dEr=dEr, dEi=dEi)
dJz = fg.deltaJz(field, dEr=dEr, dEi=dEi)
# dOAM = (dJz * W - Jz * dW) / W ** 2
dOAM = np.sqrt(dJz ** 2 / W ** 2 + Jz ** 2 * dW ** 2 / W ** 4)
print(W, dW, Jz, dJz, OAM, dOAM)
exit()
intensity = np.double(fg.readingFile(directory + f'\\intensity.mat', fieldToRead='Intensity'))
phase = fg.readingFile(directory + f'\\phase.mat', fieldToRead='phase')

fg.plot_2D(np.abs(intensity))
fg.plot_2D(phase)

shape = np.shape(intensity)
field = np.zeros(shape)
# tem = np.copy(intensity[0:50, 0:400])
# for i in range(8):
#     intensity[(50 * i):(50 * i + 50), 0:400] -= tem
for i in range(shape[0]):
    for j in range(shape[1]):
        if intensity[i, j] < 0:
            field[i, j] = - np.sqrt(-intensity[i, j])
        # elif intensity[i, j] == 0:
        #     field[i, j] = 1
        else:
            field[i, j] = np.sqrt(intensity[i, j])

field = field * np.exp(1j * phase)
fg.plot_2D(np.abs(field))
fg.plot_2D(phase)
plt.show()
fg.Jz_calc(field)
Jz = fg.Jz_calc_no_conj(field)
W = fg.W_energy(field)
OAM = Jz / W
dEr, dEi = np.real(field) * 0.1, np.imag(field) * 0.1
dEr = dEi = np.ones(np.shape(field))
# print(abs(field[:, 65]))
# exit()
dW = fg.deltaW(field, dEr=dEr, dEi=dEi)
dJz = fg.deltaJz(field, dEr=dEr, dEi=dEi)
# dOAM = (dJz * W - Jz * dW) / W ** 2
dOAM = np.sqrt(dJz ** 2 / W ** 2 + Jz ** 2 * dW ** 2 / W ** 4)
print(W, dW, Jz, dJz, OAM, dOAM)
# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)

# xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
# yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
# xArray = list(range(np.shape(field)[0]))
# yArray = list(range(np.shape(field)[1]))
# fg.Jz_calc(field, xArray, yArray)
# fg.plot_2D(np.abs(field) ** 2, axis_equal=True)
# plt.show()
