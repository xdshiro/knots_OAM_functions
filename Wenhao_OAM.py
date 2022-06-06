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
field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l =3) + fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], l =1)
# field = fOAM.LG_simple(xyzMesh[0], xyzMesh[1], xyzMesh[2], w=1.2, width=1.2, k0=1, z0=0., knot=None)

# xArray = np.linspace(-xyMinMax, xyMinMax, xRes)
# yArray = np.linspace(-xyMinMax, xyMinMax, yRes)
# xArray = list(range(np.shape(field)[0]))
# yArray = list(range(np.shape(field)[1]))
# fg.Jz_calc(field, xArray, yArray)
fg.Jz_calc(field)
# fg.plot_2D(np.abs(field) ** 2, axis_equal=True)
# plt.show()
