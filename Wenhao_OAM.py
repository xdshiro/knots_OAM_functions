import numpy as np
import matplotlib.pyplot as plt

# mine
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM
import os

xyMinMax = 5

xRes = yRes = 70
xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, 0.5, xRes, yRes, 1)
field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=-1) \
        + fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l=0)
fg.plot_2D(np.angle(field)[:, :, np.shape(field)[2] // 2])
# plt.show()
fg.Jz_calc(field)
exit()
intensity = np.double(fg.readingFile(f'.\\PhaseRetrieval 1\\intensity.mat', fieldToRead='Intensity'))
tem = np.copy(intensity[0:50, 0:400])
for i in range(8):
    intensity[(50 * i):(50 * i + 50), 0:400] -= tem
# fg.plot_2D(np.abs(intensity))
phase = fg.readingFile(f'.\\PhaseRetrieval 1\\phase.mat', fieldToRead='phase')

# plt.show()
intensity[intensity < 0] = 0
fieldAbs = np.sqrt(intensity)
print(intensity.min())
field = np.sqrt(intensity) * np.exp(1j * phase)
# fg.plot_2D(np.abs(field))
# plt.show()
# intensityNew = np.abs(intensity - np.mean(intensity[0:50, 0:400]))
# field = np.sqrt(intensityNew) * np.exp(1j * phase)
fg.Jz_calc(field)
# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)

# xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
# yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
# xArray = list(range(np.shape(field)[0]))
# yArray = list(range(np.shape(field)[1]))
# fg.Jz_calc(field, xArray, yArray)
# fg.plot_2D(np.abs(field) ** 2, axis_equal=True)
# plt.show()
