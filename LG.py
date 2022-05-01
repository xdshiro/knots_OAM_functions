from scipy.special import erf, jv, iv, assoc_laguerre
import planarity

xResolution = 356  # N_per 2 661
yResolution = 356
xStart, xFinish = 0, 3 * 1e-3  # 10
yStart, yFinish = 0, 3 * 1e-3  # 10
x0 = (xStart + xFinish) / 2
y0 = (yStart + yFinish) / 2 #+ 0j * 1e-6

rho0 = 0.5e-3

lOAM = 1

lambda0 = 500e-9
k0 = 2 * np.pi / lambda0
cSOL = 3e8
w0 = k0 * cSOL
def rayleigh_range(lambda0, rho0):
    return np.pi * rho0 ** 2 / lambda0

def asymmetric_Gauss(x, y, z, l=lOAM, p=0, width=rho0):
    zR = rayleigh_range(lambda0, width)
    x = x - x0 + 1j * 1e-2
    y = y - y0
    width = 1e-2
    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def width_z(z):
        return width * np.sqrt(1 + (z / zR) ** 2)

    def E0():
        return 1

    def R(z):
        return z * (1 + (zR / z) ** 2)

    def ksi(z):
        return np.angle(z / zR)

    return (E0() * (width / width_z(z)) * np.exp(-rho(x, y) / (width_z(z)) ** 2)
            * np.exp(-1j * (k0 * z + k0 * rho(x, y) ** 2 / 2 / R(z) - ksi(z))))


def asymmetric_LG(x, y, z, l=lOAM, p=0, width=rho0):
    x = x - x0
    y = y - y0 # + 1j * 1e-3 / 6
    zR = rayleigh_range(lambda0, width)
    print("Rayleigh Range: ", zR)
    #z = 1e-20

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def width_z(z):
        return width * np.sqrt(1 + (z / zR) ** 2)

    def R(z):
        return z * (1 + (zR / z) ** 2)

    def ksi(z):
        return np.arctan(z / zR)

    def laguerre_polynomial(x, l, p):
        return assoc_laguerre(x, p, l)

    def nonlinearity(x, y):
        # return x + 1j * np.sign(l) * y
        return (np.sqrt((x) ** 2 + y ** 2) * np.exp(1j * np.angle(x + 1j * y)))

    #
    #y = y - y0 #+ 1j * 1e-3 / 6
    E = (width_z(0) / width_z(z) * (np.sqrt(2) / width_z(z)) ** (np.abs(l))
         * nonlinearity(x, y) ** (np.abs(l))
         * laguerre_polynomial(2 * rho(x, y) ** 2 / width_z(z) ** 2, np.abs(l), p)
         * np.exp(-rho(x, y) ** 2 / width_z(z) ** 2 + 1j * k0 * rho(x, y) ** 2 / (2 * R(z))
                  - 1j * (np.abs(l) + 2 * p + 1) * ksi(z)))
    return E

def asymmetric_BG(x, y, z, c, l=lOAM, width=rho0, alphaPar=-13.):
    x = x - x0
    y = y - y0
    theta0 = 0  # ???????????????????????????????????????????
    print(x0, y0)
    #z = 1e-20
    if alphaPar == -13:
        alpha = k0 * np.sin(theta0)
    else:
        alpha = alphaPar
    print(alphaPar)
    def phi(x, y):
        return np.angle(x + 1j * y)

    def rho(x, y):
        return np.sqrt(x ** 2 + y ** 2)

    def q(z):
        z0 = rayleigh_range(lambda0, width)
        print("Rayleigh Range: ", z0)
        return 1 + 1j * z / z0

    def J(arg, l):
        return jv(l, arg)

    def asymmetry(x, y, z, c):
        return (0e0 * 2 * c * q(z) * np.exp(1j * (phi(x, y)))
                / 1)

    def extra_phase_Dima(x, y):
        return np.exp(0j* (phi(x, y) - 4 * rho(x, y) / rho0))

    print(c, alpha, l, width)
    E = ((1 / q(z))
         * np.exp(1j * k0 * z
                  - 1j * alpha ** 2 * z / (2 * k0 * q(z))
                  - rho(x, y) ** 2 / (q(z) * width ** 2)
                  + 1j * l * phi(x, y))
         * (alpha * rho(x, y)
            / (alpha * rho(x, y) - asymmetry(x, y, z, c))) ** (l / 2)
         * J(np.sqrt(alpha * rho(x, y) * (alpha * rho(x, y) - asymmetry(x, y, z, c))) / q(z), l)
         * extra_phase_Dima(x, y))
    return E

# print("Rayleigh length: ", zR, " Kerr Collapse length: ", Lcollapse())

# %% Arrays creation
xArray = np.linspace(xStart, xFinish, xResolution)
yArray = np.linspace(yStart, yFinish, yResolution)
xyMesh = np.array(np.meshgrid(xArray, yArray, indexing='ij'))
kxArray = np.linspace(-1. * np.pi * (xResolution - 2) / xFinish,
                      1. * np.pi * (xResolution - 2) / xFinish, xResolution)
kyArray = np.linspace(-1. * np.pi * (yResolution - 2) / yFinish,
                      1. * np.pi * (yResolution - 2) / yFinish, yResolution)
KxywMesh = np.array(np.meshgrid(kxArray, kyArray, indexing='ij'))

def Jz_calc(EArray, xArray, yArray):
    x0 = (xArray[-1] + xArray[0]) / 2
    y0 = (yArray[-1] + yArray[0]) / 2
    sum = 0
    dx = xArray[1] - xArray[0]
    dy = yArray[1] - yArray[0]
    x = xArray - x0
    y = yArray - y0
    for i in range(1,len(xArray)-1,1):
        for j in range(1,len(yArray)-1,1):
            dEx = (EArray[i + 1, j] - EArray[i -1, j]) / (2 * dx)
            dEy = (EArray[i, j + 1] - EArray[i, j - 1]) / (2 * dy)
            sum += (np.conj(EArray[i, j]) *
                    (x[i] * dEy - y[j] * dEx))

    return np.imag(sum * dx * dy)

def plot_2D(E, x, y, xname='', yname='', map='jet', vmin=0.13, vmax=1.14, title='', ticksFontSize=14,
    xyLabelFontSize=14, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    if vmin == 0.13 and vmax == 1.14:
        vmin = E.min()
        vmax = E.max()

    image = plt.imshow(E,
                       interpolation='bilinear', cmap=map,
                       origin='lower', aspect='auto',  # aspect ration of the axes
                       extent=[y[0], y[-1], x[0], x[-1]],
                       vmin=vmin, vmax=vmax, label='sdfsd')
    cbr = plt.colorbar(image, shrink=1, pad=0.02, fraction=0.1)
    cbr.ax.tick_params(labelsize=ticksFontSize)
    plt.xticks(fontsize=ticksFontSize)
    plt.yticks(fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize)
    ax.set_ylabel(yname, fontsize=xyLabelFontSize)
    plt.title(title, fontweight="bold", fontsize=26)
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
    plt.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    z = 1e-6
    fieldOLD = (1.51 * asymmetric_LG(xyMesh[0], xyMesh[1], z, l=0, p=0, width=rho0) +
                -5.06 * asymmetric_LG(xyMesh[0], xyMesh[1], z, l=0, p=1, width=rho0) +
                7.23 * asymmetric_LG(xyMesh[0], xyMesh[1], z, l=0, p=2, width=rho0) +
                -2.03 * asymmetric_LG(xyMesh[0], xyMesh[1], z, l=0, p=3, width=rho0) +
                -3.97 * asymmetric_LG(xyMesh[0], xyMesh[1], z, l=3, p=0, width=rho0))


    plot_2D(np.abs(fieldOLD), xArray, yArray)
    plot_2D(np.angle(fieldOLD), xArray, yArray, map='hsv')