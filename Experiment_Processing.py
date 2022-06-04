import numpy as np
import matplotlib.pyplot as plt

# mine
import functions_general as fg
import functions_OAM_knots as fOAM


# from knots_optimization import *


def read_file_experiment_knot(_file_name, save_file=None, plot=False, **kwargs):
    field_experiment = fg.readingFile('all_other_data/experiment/_file_name.mat', fieldToRead='Uz',
                                      printV=False)
    # field_experiment = field_experiment[::resDec, ::resDec]

    # fg.plot_2D(np.abs(field_experiment[::resDec, ::resDec]))
    # fg.plot_2D(np.angle(field_experiment[::resDec, ::resDec]))
    # plt.show()
    # exit()
    # fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=15, stepsNumber=40, n0=1, k0=1)
    newField = region_increase(field_experiment, **kwargs)
    if plot:
        fg.plot_2D(np.abs(newField[:, :]))
        fg.plot_2D(np.angle(newField[:, :]))
        plt.show()
    if save_file is not None:
        np.save(save_file, newField)
    return newField


fileName = f'all_other_data/experiment/3foil_field2_modIntensity_f.mat'
plot = False
resNew = 40
res_increase = 1.5
field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
                                  printV=False)
field_experiment_inter = fg.field_new_resolution(field_experiment, resNew, resNew)
newField = region_increase(field_experiment_inter, 5, xy_increase=res_increase,
                           xy_res_increase=res_increase, k_increase=6,
                           plot_origian=False, plot_new=False, k_res_increase=res_increase, cut=False)
if plot:
    fg.plot_2D(np.abs(newField[:, :]))
    fg.plot_2D(np.angle(newField[:, :]))
    plt.show()
fieldAfterProp = fg.one_plane_propagator(newField, dz=1.1, stepsNumber=23, n0=1, k0=1)
fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 4, circle=True)
fOAM.plot_knot_dots(fieldAfterProp, axesAll=True, size=250, color='b')
if plot:
    plt.show()
