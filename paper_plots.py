import matplotlib.pyplot as plt
import functions_general as fg


# figure OAM vs KNOTS comparison (vs turbulence power) 1D
def plot_1():
    xArray = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [92.56198347, 78.92561983, 68.87052342, 66.11570248,
              65.84022039, 61.98347107, 50.41322314, 23.96694215,
              4.958677686]
    fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(xArray, yArray, color='r', marker='o', label='Trefoil',
               xname='Strehl ratio', yname='Recovered knots %',
               loc='lower left', ax=ax)
    xArray = [0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [100, 95.8677686, 73.14049587, 42.14876033, 37.60330579, 30.16528926,
              19.83471074, 12.39669421]
    fg.plot_1D(xArray, yArray, color='dodgerblue', marker='o', label=r'OAM l=3',
               xname='Strehl ratio', yname='Recovered knots/LG beams %',
               loc='lower left', ax=ax)
    ax.invert_xaxis()
    # plt.xlim(0.021, -.0007)
    plt.ylim(0, 102.5)
    # ax.set_xscale('log')
    ax.fill_between([0.0018, 0.0175], [0, 0], [102.5, 102.5],
                    facecolor='g',
                    alpha=0.3,
                    color='green',
                    edgecolor='black',
                    linewidth=1,
                    linestyle='--')
    # plt.xticks([0.02, 0.01, 0.001])
    # fig.set_figwidth(12)
    # fig.set_figheight(6)
    # fig.set_facecolor('floralwhite')
    # ax.set_facecolor('seashell')

    plt.show()


def plot_percentage_vs_w(ax=None):
    xArray = [1.05, 1.075, 1.1, 1.125, 1.15, 1.175, 1.2, 1.3, 1.4]
    yArray = [1.652892562, 29.75206612, 38.84297521, 36.36363636, 32.23140496,
              28.09917355, 23.96694215, 14.87603306, 10.74380165]
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(xArray, yArray, color='r', marker='o', label='Trefoil',
               xname='Strehl ratio', yname='Recovered knots %',
               loc='upper left', ax=ax, title='SR=0.001 (v2)')
    # ax.invert_xaxis()
    plt.xlim(1, 1.5)
    plt.ylim(0, 100)
    # ax.text('SR=0.001')
    # ax.set_xscale('log')
    # ax.fill_between([0.0018, 0.0175], [0, 0], [102.5, 102.5],
    #                 facecolor='g',
    #                 alpha=0.3,
    #                 color='green',
    #                 edgecolor='black',
    #                 linewidth=1,
    #                 linestyle='--')
    # plt.xticks([0.02, 0.01, 0.001])
    # fig.set_figwidth(12)
    # fig.set_figheight(6)
    # fig.set_facecolor('floralwhite')
    # ax.set_facecolor('seashell')
    return ax


plot_percentage_vs_w()
plt.show()
