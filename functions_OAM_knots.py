import numpy as np
from scipy.special import assoc_laguerre
import functions_general as fg


def laguerre_polynomial(x, l, p):
    return assoc_laguerre(x, p, l)


def LG_simple(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0):
    x = x - x0
    y = y - y0
    zR = k0 * width ** 2

    E = (np.sqrt(np.math.factorial(p) / (np.pi * np.math.factorial(np.abs(l) + p)))
         * fg.rho(x, y) ** np.abs(l) * np.exp(1j * l * fg.phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-fg.rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(fg.rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    return E


def link(x, y, z, w, width=1, k0=1, z0=0.):
    a00 = -1 + w ** 2
    a01 = -w ** 2
    a10 = -2 * w
    z = z - z0

    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a10 * LG_simple(x, y, z, l=1, p=0, width=width, k0=k0))

    return field


def trefoil(x, y, z, w, width=1, k0=1, z0=0.):
    z = z - z0
    a00 = 1 - w ** 2 - 2 * w ** 4 + 6 * w ** 6
    a01 = w ** 2 * (1 + 4 * w ** 2 - 18 * w ** 4)
    a02 = - 2 * w ** 4 * (1 - 9 * w ** 2)
    a03 = -6 * w ** 6
    a30 = -8 * np.sqrt(6) * w ** 3
    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a02 * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             a03 * LG_simple(x, y, z, l=0, p=3, width=width, k0=k0) +
             a30 * LG_simple(x, y, z, l=3, p=0, width=width, k0=k0)
             )

    return field


def trefoil_mod(x, y, z, w, width=1, k0=1, z0=0.):
    z = z - z0
    H = 1.0
    a00 = 1 * (H ** 6 - H ** 4 * w ** 2 - 2 * H ** 2 * w ** 4 + 6 * w ** 6) / H ** 6
    a01 = (w ** 2 * (1 * H ** 4 + 4 * w ** 2 * H ** 2 - 18 * w ** 4)) / H ** 6
    a02 = (- 2 * w ** 4 * (H ** 2 - 9 * w ** 2)) / H ** 6
    a03 = (-6 * w ** 6) / H ** 6
    a30 = (-8 * np.sqrt(6) * w ** 3) / H ** 3
    modified = True
    if modified:
        a00 = 1.51
        a01 = -5.06
        a02 = 7.23
        a03 = -2.03
        a30 = -3.97

    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a02 * LG_simple(x, y, z, l=0, p=2, width=width, k0=k0) +
             a03 * LG_simple(x, y, z, l=0, p=3, width=width, k0=k0) +
             a30 * LG_simple(x, y, z, l=3, p=0, width=width, k0=k0)
             )

    return field


def knot_all(x, y, z, w, width=1, k0=1, z0=0., knot=None):
    if knot is None:
        knot = 'trefoil'
    if knot == 'trefoil':
        return trefoil(x, y, z, w, width=width, k0=k0, z0=z0)
    elif knot == 'hopf':
        print('add the HOPF!')
    elif knot == 'link':
        return link(x, y, z, w, width=width, k0=k0, z0=z0)
    elif knot == 'trefoil_mod':
        return trefoil_mod(x, y, z, w, width=width, k0=k0, z0=z0)


# this one hasn't been checked yet
def Jz_calc(EArray, xArray, yArray):
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    sum = 0
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x = xArray - x0
    y = yArray - y0
    for i in range(1, len(xArray) - 1, 1):
        for j in range(1, len(yArray) - 1, 1):
            dEx = (EArray[i + 1, j] - EArray[i - 1, j]) / (2 * dx)
            dEy = (EArray[i, j + 1] - EArray[i, j - 1]) / (2 * dy)
            sum += (np.conj(EArray[i, j]) *
                    (x[i] * dEy - y[j] * dEx))

    return np.imag(sum * dx * dy)


# ploting and saving math trefoil
def knot_field_plot_save(xyMax=3, zMax=1, xyRes=50, zRes=50, w=1, width=1, k0=1,
                         knot=None, axis_equal=True,
                         save=False, saveName='rename_me', plot=True, plotLayer=None):
    xyArray = np.linspace(-xyMax, xyMax, xyRes)
    xyzMesh = fg.create_mesh_XYZ(xMin=-xyMax, xMax=xyMax,
                                 yMin=-xyMax, yMax=xyMax,
                                 zMin=-zMax, zMax=zMax, xRes=xyRes, yRes=xyRes, zRes=zRes)
    field = knot_all(xyzMesh[0], xyzMesh[1], z=xyzMesh[2], w=w, width=width, k0=k0, knot=knot)
    if plot:
        if plotLayer is None:
            plotLayer = zRes // 2
        fg.plot_2D(np.abs(field)[:, :, plotLayer], xyArray, xyArray, axis_equal=axis_equal)
        fg.plot_2D(np.angle(field)[:, :, plotLayer], xyArray, xyArray, map='hsv', axis_equal=axis_equal)
    if save:
        np.save(saveName, field)


def milnor_Pol_testing(x, y, z, *args):
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-1 ** 2 + R ** 2 + 2j * z * 1 + z ** 2) / (1 ** 2 + R ** 2 + z ** 2)
    v = (2 * R * 1 * np.exp(1j * f)) / (1 ** 2 + R ** 2 + z ** 2)
    divider = (1 + R ** 2 + z ** 2)
    return u ** 3 - v / 10 #/ divider ** 2
    # return (
    #         (-2 * np.exp(1j * f) * R * (1 + R ** 2 + z ** 2) ** 2
    #          + (-1 + R ** 2 + 2 * 1j * z + z ** 2) ** 5)
    #         / (1 + R ** 2 + z ** 2) ** 5
    # )


# def milnor_Pol_testing(x, y, z):
#     R = fg.rho(x, y)
#     f = fg.phi(x, y)
#     return (
#             (-2 * np.exp(1j * f) * R * (1 + R ** 2 + z ** 2) ** 2
#              + (-1 + R ** 2 + 2 * 1j * z + z ** 2) ** 5)
#             / (1 + R ** 2 + z ** 2) ** 3
#     )


def milnor_Pol_u_v_any(x, y, z, uOrder, vOrder, H=1):
    """This function create u^a-v^b Milnor polynomial"""
    R = fg.rho(x, y)
    f = fg.phi(x, y)
    u = (-H ** 2 + R ** 2 + 2j * z * H + z ** 2) / (H ** 2 + R ** 2 + z ** 2)
    v = (2 * R * H * np.exp(1j * f)) / (H ** 2 + R ** 2 + z ** 2)
    return u ** uOrder - v ** vOrder
