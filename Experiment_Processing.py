import numpy as np
import matplotlib.pyplot as plt

# mine
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM
import os


# from knots_optimization import *


def read_file_experiment_knot(fileName):
    field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
                                      printV=False)
    print(np.shape(field_experiment))
    fg.plot_2D(np.abs(field_experiment) ** 2, axis_equal=True)
    fg.plot_2D(np.angle(field_experiment), axis_equal=True)
    fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=1.5*5, stepsNumber=30, n0=1, k0=1)
    fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 3.5, circle=True)
    fOAM.plot_knot_dots(fieldAfterProp, axesAll=False, size=250, color='b')
    plt.show()


def create_the_folder_from_experiment(directory, directorySave):
    listOfFiles = [f for f in os.listdir(directory)]
    for num, files in enumerate(listOfFiles):
        files = files.replace('.mat', '')
        print(num)
        field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
                                          printV=False)
        fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=1.1 * 6, stepsNumber=22, n0=1, k0=1)
        fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 4, circle=True)
        # fOAM.plot_knot_dots(fieldAfterProp, axesAll=True, size=250, color='b')

        if not os.path.exists(directorySave):
            os.makedirs(directorySave)
        fOAM.save_knot_dots(fieldAfterProp, directorySave + files)
read_file_experiment_knot('3foil_field2_modIntensity_f.mat')
exit()
# directory = 'C:\\Users\\Dima\\Box\\Knots Exp\\DATA_FINAL\\Trefoil Exp\\SR = 0.9\\'
# directorySave = '.\\experiment\\trefoil\\SR = 0.9\\'
# create_the_folder_from_experiment(directory, directorySave)
# exit()
# name = '3foil_turb_SR_9.000000e-01_num_2.mat'
# fileName = directory + name
plot2D = True
save = False
resNew = 80
res_increase = 1.5
#
#
fileName = '3foil_noturb (8).mat'
field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
                                  printV=False)

# newField = field_experiment
if save:
    field_experiment_inter = fg.field_new_resolution(field_experiment, resNew, resNew)
    newField = fg.region_increase(
        field_experiment_inter, 5, xy_increase=res_increase,
        xy_res_increase=res_increase, k_increase=6,
        plot_origian=False, plot_new=False, k_res_increase=res_increase, cut=False
    )
    np.save('test_delete8', newField)

newField = np.load('test_delete.npy')
# newField = field_experiment
# newField = np.load('test_delete.npy', allow_pickle=True).item()
if plot2D:
    fg.plot_2D(np.abs(newField[:, :]))
    fg.plot_2D(np.angle(newField[:, :]))
    plt.show()
fieldAfterProp = fg.one_plane_propagator(newField, dz=2, stepsNumber=30, n0=1, k0=1)
fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 4, circle=True)
fOAM.plot_knot_dots(fieldAfterProp, axesAll=True, size=250, color='b')
plt.show()
