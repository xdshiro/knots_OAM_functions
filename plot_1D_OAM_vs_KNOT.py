import matplotlib.pyplot as plt
import general_functions as gf

if __name__ == '__main__':
    xArray = [0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [92.56198347, 78.92561983, 68.87052342, 66.11570248,
              65.84022039, 61.98347107, 50.41322314, 23.96694215,
              4.958677686]
    fig, ax = plt.subplots()  # figsize=(8, 6)
    gf.plot_1D(xArray, yArray, color='r', marker='o', label='Trefoil',
               xname='Strehl ratio', yname='Recovered knots %',
               loc='lower left', ax=ax)
    xArray = [0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    yArray = [100, 95.8677686, 73.14049587, 42.14876033, 37.60330579, 30.16528926,
              19.83471074, 12.39669421]
    gf.plot_1D(xArray, yArray, color='dodgerblue', marker='o', label=r'OAM l=3',
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
