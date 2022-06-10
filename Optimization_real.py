"""
This module optimize the knot in turbulence based on the region in N-D amplitudes space.
Finds the "knot-amplitude-space" and find the centre of it based on the weight coefficients
"""
import my_modules.functions_OAM_knots as fOAM
import my_modules.functions_general as fg
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    xyMinMax = 4  # * 2
    zMinMax = 0.9  # * 4
    zRes = 40
    xRes = yRes = 40
    plot_test = True
    xyzMesh = fg.create_mesh_XYZ(xyMinMax, xyMinMax, zMinMax, xRes, yRes, zRes, zMin=None)

    field = fOAM.trefoil_mod(*xyzMesh, w=1.6, width=1, k0=1, z0=0., coeff=None, coeffPrint=False)

    fg.plot_2D(np.abs(field[:, :, 35]))
    fOAM.plot_knot_dots(field)
    fg.plot_3D_density(np.angle(field))
    plt.show()

