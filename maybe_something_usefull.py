import numpy as np
from scipy.optimize import fsolve


def func(x):
    h = 5e-3
    return 1 * x * h * np.tan(x * h) - h * np.sqrt(335.1 ** 2 - x ** 2)


def func2(x):
    h = 5e-3
    return -.2 * x * h * np.tan(x * h) ** (-1) - h * np.sqrt(335.1 ** 2 - x ** 2)


tau_initial_guess = 200
print(fsolve(func, tau_initial_guess))
tau_initial_guess = 350
print(fsolve(func2, tau_initial_guess))

exit()
from scipy.special import assoc_laguerre
from scipy import integrate


def complexIntegral(f, x1, x2, **kwargs):

    def real(x):
        return np.real(f(x))

    def imag(x):
        return np.imag(f(x))

    real_integral = integrate.quad(real, x1, x2, **kwargs)
    imag_integral = integrate.quad(imag, x1, x2, **kwargs)
    return real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:]

def phi_classic(x, y):
    if x == 0:
        if y >= 0:
            return np.pi / 2
        else:
            return 3 * np.pi / 2
    if x > 0:
        if y >= 0:
            return np.arctan(y/x)
        else:
            return 2 * np.pi + np.arctan(y/x)
    else:
        return np.pi + np.arctan(y/x)


def phi(x, y):
    return np.angle(x + 1j * y)


def rho(x, y):
    return np.sqrt(x ** 2 + y ** 2)

def f(x, y, z):
    return 4 * np.exp(1j * 2 * phi(x, y)) + 1 * np.exp(1j * 1 * phi(x, y))


def integral(f, r, z, l):

    def y(x, sign):
        return sign * np.sqrt(r ** 2 - x ** 2)

    def f1(x):
        Y = y(x, +1)
        return f(x, Y, z) * (-1) * Y / r ** 2 * np.exp(-1j * l * phi(x, Y))

    def f2(x):
        Y = y(x, -1)
        return f(x, Y, z) * (-1) * Y / r ** 2 * np.exp(-1j * l * phi(x, Y))

    i1 = complexIntegral(f1, r, -r)
    i2 = complexIntegral(f2, -r, r)

    answer = i1[0] + i2[0]
    return answer





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(integral(f, 1, 10, 4))
    #t = sympy.symbols("t")
    #print(scipy.special.genlaguerre(4, 0))
    text = 'Never forget what you are, for surely the world will not'
    print("Index of N:", text.find("N"), "\nIndex of ,:", text.find(","))
    print(f"Index of N: {text.find('N')}\nIndex of ,: {text.find(',')}")
    print('a', 'b')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import numpy as np
import matplotlib.pyplot as plt

global fig, ax
ticksFontSize = 18
xyLabelFontSize = 20
legendFontSize = 20


def plot_1D(x, y, label=None, xname='', yname='', ls='-', lw=4, color=None,
            ticksFontSize=ticksFontSize, xyLabelFontSize=xyLabelFontSize,
            legendFontSize=legendFontSize,
            marker='', markersize=12, markeredgecolor='k',
            loc='lower right'):
    ax.plot(x, y, linestyle=ls,
            marker=marker, markersize=markersize, markeredgecolor=markeredgecolor,
            label=label, lw=lw, color=color)
    """    if dotsColor == 'rrr':
        plt.scatter(x, y, s=dotsSize, label=False)
    else:
        plt.scatter(x, y, s=dotsSize, label=False, color=color)"""
    plt.xticks(fontname='Times New Roman', fontsize=ticksFontSize)
    plt.yticks(fontname='Times New Roman', fontsize=ticksFontSize)
    ax.set_xlabel(xname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    ax.set_ylabel(yname, fontsize=xyLabelFontSize, fontname='Times New Roman')
    # ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    if label:
        legend = ax.legend(shadow=True, fontsize=legendFontSize,
                  facecolor='white', edgecolor='black', loc=loc)
        plt.setp(legend.texts, family='Times New Roman')


if __name__ == '__main__':

    xArray = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [92.56198347, 78.92561983, 68.87052342, 66.11570248,
              65.84022039, 61.98347107, 50.41322314, 23.96694215,
              4.958677686]
    fig, ax = plt.subplots()  # figsize=(8, 6)
    plot_1D(xArray, yArray, color='r', marker='o', label='Trefoil',
            xname='Strehl ratio', yname='Recovered knots %',
            loc='lower left'
            )
    xArray = [0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [100, 95.8677686, 73.14049587, 42.14876033, 37.60330579, 30.16528926,
              19.83471074, 12.39669421]
    plot_1D(xArray, yArray, color='dodgerblue', marker='o', label=r'OAM l=3',
            xname='Strehl ratio', yname='Recovered knots/LG beams %',
            loc='lower left')
    ax.invert_xaxis()
    # plt.xlim(0.021, -.0007)
    plt.ylim(0, 102.5)
    #ax.set_xscale('log')
    ax.fill_between([0.0018, 0.0175], [0, 0], [102.5, 102.5],
                    facecolor='g',
                    alpha=0.3,
                    color='green',
                    edgecolor='black',
                    linewidth=1,
                    linestyle='--')
    # plt.xticks([0.02, 0.01, 0.001])
    #fig.set_figwidth(12)
    #fig.set_figheight(6)
    # fig.set_facecolor('floralwhite')
    #ax.set_facecolor('seashell')

    plt.show()




W = (np.sum(np.conj(fieldOLD) * fieldOLD, axis=0)
             * (xArray[1] - xArray[0]) * (yArray[1] - yArray[0]))
        W = np.sum(W, axis=0)
        for i in zparM2:
            Jz = Jz_calc(fieldOLD[:, :, i], xArray, yArray)
            print(W[i], Jz / W[i])