import matplotlib.pyplot as plt
import functions_general as fg

"""
HOPF:
w_dep_SR=0.95(1.3) w_0=1.3
w=1.525 28; w=1.475 32; w=1.4 12; w=1.45 30; w=1.5 31; w=1.55 25; w=1.6 21; w=1.65 19;

w=1.475, w_0=1.3 SR dependence
SR=0.6 0; SR=0.7 0; SR=0.75 0; SR=0.8 0; SR=0.85 2; SR=0.9 13; SR=0.95 32;

Best w_0=1.3 SR dependence 2.96*LG(0,0,z) -6.23*LG(1,0,z) +4.75*LG(2,0,z) +5.49*LG(0,2,z); %12 + 1313
SR=0.6 0; SR=0.7 0; SR=0.75 3; SR=0.8 12; SR=0.85 17; SR=0.9 34; SR=0.95 49;

Old our best
yArray = [65, 27, 7, 3] xArray=[0.95, 0.9, 0.8, 0.7]

Paper Hopf
SR=0.6 ; SR=0.7 ; SR=0.75 ; SR=0.8 ; SR=0.85 1; SR=0.9 3; SR=0.95 17;


TREFOIL:
Best w_0=1.3 SR dependence; % 12 dots best 
% an_01 = -4.1911; % an_02 = 7.9556; % an_03 = -3.4812; % an_30 = -4.2231;
SR=0.6 1 5 94; SR=0.7 4 11 85; SR=0.75 7 24 69; SR=0.8 7 23 70; SR=0.85 16 31 53; SR=0.9 42 33 25; SR=0.95 72 22 6;

OLD w_0=1.3? % 12 dots best 
SR=0.6 0; SR=0.7 0; SR=0.75 4; SR=0.8 8; SR=0.85 17; SR=0.9 37; SR=0.95 84;

avg: 1 + 2 
SR=0.6 0.5 4.5 95; SR=0.7 2 12 86; SR=0.75 5.5 22.5 72; SR=0.8 7.5 23.5 69;
 SR=0.85 16.5 30.5 53; SR=0.9 39.5 33 27.5; SR=0.95 78 19 3;
modified: SR=0.6 0.5 4.5 95; SR=0.7 2 12 86; SR=0.75 5.5 20.5 74; SR=0.8 7.5 25.5 67;
 SR=0.85 16.5 30.5 53; SR=0.9 39.5 33 27.5; SR=0.95 78 19 3;
 
 Nodal lines
"""

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

def plot_percentage_Hopf_vs_w(ax=None):
    xArray = [1.375, 1.4, 1.45, 1.475, 1.5, 1.525, 1.55, 1.6, 1.65]
    yArray = [0, 12, 30, 33, 31, 28, 25, 21, 19]  # 6.67
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(
        xArray, yArray, color='b', marker='o', label=None,
        xname='coefficient w', yname='Recovered Hopf knots %',
        loc='upper left', ax=ax, title='SR=0.95'
    )
    # ax.invert_xaxis()
    plt.xlim(1.355, 1.67)
    plt.ylim(-1, 35)

    return ax


def plot_percentage_trefoil_vs_w(ax=None):
    xArray = [1, 1.05, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]
    yArray = [0, 4, 6.75, 8, 7, 5.5, 4, 2, 1, 0.67, 0.33]  # 6.67
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(
        xArray, yArray, color='r', marker='o', label=None,
        xname='coefficient w', yname='Recovered knots %',
        loc='upper left', ax=ax, title='SR=0.95'
    )
    # ax.invert_xaxis()
    plt.xlim(0.98, 1.62)
    plt.ylim(-.2, 8.5)
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


def plot_percentage_vs_w(ax=None):
    xArray = [1, 1.05, 1.1, 1.125, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6]
    yArray = [0, 4, 6.75, 8, 7, 5.5, 4, 2, 1, 0.67, 0.33]  # 6.67
    xArrayH = [1.375, 1.4, 1.45, 1.475, 1.5, 1.525, 1.55, 1.6, 1.65]
    yArrayH = [0, 12, 30, 33, 31, 28, 25, 21, 19]
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(
        xArray, yArray, color='r', marker='o', label='Trefoil',
        xname='coefficient w', yname='Recovered knots %',
        loc='upper left', ax=ax, title='SR=0.95',
    )
    fg.plot_1D(
        xArrayH, yArrayH, color='b', marker='o', label='Hopf',
        xname='coefficient w', yname='Recovered knots %',
        loc='upper left', ax=ax, title='SR=0.95',
    )
    # ax.invert_xaxis()
    # plt.xlim(0.98, 1.62)
    # plt.ylim(-.2, 7)
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


def plot_SR_percentage_Trefoil(ax=None):

    xArray = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]  # our
    yArray = [78, 39.5, 16.5, 7.5, 5.5, 2, 0.5]
    xArray2 = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6]  # pap
    yArray2 = [34, 8.5, 4, 1, 0, 0]  # for 0.9 was 5, now 12;  !!!!! 0.95 check
    xArray2 = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]  # pap without error
    yArray2 = [41, 16, 4.5, 2, 0, 0]  # for 0.95 was 39
    # yArray2 = [34, 12, 4, 1, 0, 0]  # for 0.9 was 5, now 12;  !!!!! 0.95 check
    xArray3 = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]  # 2
    yArray3 = [8, 4, 2, 0, 0, 0]  # 9 5 2
    xExp_opt = [0.95, 0.92, 0.9, 0.89, 0.85, 0.8, 0.75, 0.7]  # 2
    yExp_opt = [71, 52, 32, 27.5, 10, 6, 2, 0]
    # xExp_opt = [0.9, 0.8, 0.75, 0.85, 0.89, 0.92, 0.85_2, 0.75_2]  # 2
    # yExp_opt = [31, 6, 0, 6, 27.5, 52, 10, 0.5]
    # xExp_pap = [0.95, 0.9, 0.89, 0.85, 0.9]  # 2 real
    # yExp_pap = [52, 15, 21, 1, 5]
    xExp_pap = [0.95, 0.9, 0.85] # [0.95, 0.9, 0.85]  # [0.95, 0.95]
    yExp_pap = [42, 13.5, 1]# [52, 13.5, 1]  # [46, 39] 0.85: 1, 0.0?
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    # ax.scatter(xExp_opt, yExp_opt)
    fg.plot_1D(
        xArray, yArray, color='r', marker='o', label='Optimized',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title=None
    )
    fg.plot_1D(
        xArray2, yArray2, color='b', marker='o', label='Dennis',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title='Trefoil'
    )
    fg.plot_1D(
        xArray3, yArray3, color='g', marker='o', label='Math_optimal',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title='Trefoil'
    )
    fg.plot_1D(
        xExp_opt, yExp_opt, color='lime', marker='>', markeredgecolor='r', lw=2, markersize=12,
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax
    )
    fg.plot_1D(
        xExp_pap, yExp_pap, color='lime', marker='<', markeredgecolor='b', lw=2, markersize=12,
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax
    )
    # for x, y in zip(xExp_opt, yExp_opt):
    #     fg.plot_1D(
    #         x, y, color='r', marker='>', markersize=15,
    #         xname='Strehl ratio', yname='Recovered knots %',
    #         loc='upper right', ax=ax
    #     )
    #
    # for x, y in zip(xExp_pap, yExp_pap):
    #     fg.plot_1D(
    #         x, y, color='b', marker='<', markersize=15,
    #         xname='Strehl ratio', yname='Recovered knots %',
    #         loc='upper right', ax=ax
    #     )
    # plt.scatter([0.9], [57], c='y', marker='o', s=80)
    # plt.scatter([0.95], [65, 53, 60], c='k', marker='x', s=80)
    # ax.invert_xaxis()
    plt.xlim(0.97, 0.68)
    plt.ylim(-0.1, 100)
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


def plot_SR_percentage_hopf(ax=None):
    #  27 (0.95)-> 13(0.9)
    xArray = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0]  # our
    yArray = [57, 30.5, 17, 9.5, 3, 1.5, 0]
    xArray2 = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]  # pap
    yArray2 = [17, 3, 1, 0,0,0,0]
    xArrayMath = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]  # math 1.475
    yArrayMath = [32, 13, 2, 0,0,0,0]
    xExp_opt = [0.95, 0.9, 0.85]  # 2
    yExp_opt = [27, 13, 4]
    xExp_opt2 = [0.95, 0.9, 0.85]  # 2
    yExp_opt2 = [51, 17, 3]
    xExp_math = [0.9]  # 2
    yExp_math = [7]
    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(
        xArray, yArray, color='r', marker='o', label='Optimized',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title=None
    )
    fg.plot_1D(
        xArray2, yArray2, color='b', marker='o', label='Dennis',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title=None
    )
    fg.plot_1D(
        xArrayMath, yArrayMath, color='g', marker='o', label='Math_optimal',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title='Hopf'
    )
    fg.plot_1D(
        xExp_opt, yExp_opt, color='k', marker='>', markeredgecolor='r', lw=2, markersize=12,
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax
    )
    fg.plot_1D(
        xExp_opt2, yExp_opt2, color='k', marker='>', markeredgecolor='r', lw=2, markersize=16,
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax
    )
    fg.plot_1D(
        xExp_math, yExp_math, color='g', marker='<', markeredgecolor='r', lw=2, markersize=12,
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax
    )
    # plt.scatter([0.9], [57], c='y', marker='o', s=80)
    # plt.scatter([0.95], [65, 53, 60], c='k', marker='x', s=80)
    # ax.invert_xaxis()
    plt.xlim(0.97, 0.68)
    plt.ylim(-3, 100)
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

def plot_SR_percentage_hopf_trefoil_comp(ax=None):
    xArrayH = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0]  # our
    yArrayH = [57, 30.5, 17, 9.5, 3, 1.5, 0]
    xArrayTr = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]  # our
    yArrayTr = [78, 39.5, 16.5, 7.5, 5.5, 2, 0.5]

    # xArray = [1.05, 1.075, 1.1, 1.125, 1.2, 1.3]
    # yArray = [0, 4.132231405, 24.79338843, 14.87603306, 20.66115702, 2.479338843]
    if ax is None:
        fig, ax = plt.subplots()  # figsize=(8, 6)
    fg.plot_1D(
        xArrayTr, yArrayTr, color='r', marker='o', label='Optimized Trefoil',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title=None
    )
    fg.plot_1D(
        xArrayH, yArrayH, color='b', marker='o', label='Optimized Hopf',
        xname='Strehl ratio', yname='Recovered knots %',
        loc='upper right', ax=ax, title=None
    )
    # plt.scatter([0.9], [57], c='y', marker='o', s=80)
    # plt.scatter([0.95], [65, 53, 60], c='k', marker='x', s=80)
    # ax.invert_xaxis()
    plt.xlim(0.97, 0.68)
    plt.ylim(-3, 100)
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
plot_SR_percentage_Trefoil(ax=None)
# plot_percentage_vs_w()
# plot_percentage_trefoil_vs_w()
# plot_percentage_Hopf_vs_w(ax=None)
plot_SR_percentage_hopf()
# plot_SR_percentage_hopf_trefoil_comp()
plt.show()
