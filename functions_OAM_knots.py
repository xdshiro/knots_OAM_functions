import numpy as np
from scipy.special import assoc_laguerre


def laguerre_polynomial(x, l, p):
    return assoc_laguerre(x, p, l)


def LG_simple(x, y, z, l=1, p=0, width=1, k0=1, x0=0, y0=0):
    x = x - x0
    y = y - y0
    zR = k0 * width ** 2
    print("Rayleigh Range: ", zR, f"k0={k0}")

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def phi(x, y):
        return np.angle(x + 1j * y)

    E = (np.sqrt(np.math.factorial(p) / (np.pi * np.math.factorial(np.abs(l) + p)))
         * rho(x, y) ** np.abs(l) * np.exp(1j * l * phi(x, y))
         / (width ** (np.abs(l) + 1) * (1 + 1j * z / zR) ** (np.abs(l) + 1))
         * ((1 - 1j * z / zR) / (1 + 1j * z / zR)) ** p
         * np.exp(-rho(x, y) ** 2 / (2 * width ** 2 * (1 + 1j * z / zR)))
         * laguerre_polynomial(rho(x, y) ** 2 / (width ** 2 * (1 + z ** 2 / zR ** 2)), np.abs(l), p)
         )
    return E


def actual_link(x, y, z, w, width=1, k0=1, z0=0.):
    a00 = -1 + w ** 2
    a01 = -w ** 2
    a10 = -2 * w
    z = z - z0

    field = (a00 * LG_simple(x, y, z, l=0, p=0, width=width, k0=k0) +
             a01 * LG_simple(x, y, z, l=0, p=1, width=width, k0=k0) +
             a10 * LG_simple(x, y, z, l=1, p=0, width=width, k0=k0))

    return field


def actual_trefoil(x, y, z, w, width=1, k0=1, z0=0.):
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