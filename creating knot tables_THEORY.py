import knot_class as kc
import functions_general as fg
import functions_OAM_knots as fOAM
import numpy as np
import matplotlib.pyplot as plt

# tableName = f'C:\\Users\\Dima\\Box\\Knots Exp\\Experimental Data\\dots\\trefoil\\3foil SR = 0.95'
# directoryName = f'C:\\Users\\Dima\\Box\\Knots Exp\\Experimental Data\\dots\\trefoil\\3foil SR = 0.95\\'
# kc.creat_knot_table_dict_dots(directoryName, tableName)
# exit()
#
#
# def read_file_experiment_knot(fileName):
#     # fileName = '3foil_field2_modIntensity_f.mat'
#     field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
#                                       printV=False)
#     fg.plot_2D(np.abs(field_experiment) ** 2, axis_equal=True)
#     fg.plot_2D(np.angle(field_experiment), axis_equal=True)
#     plt.show()
#     # exit()
#     # field_experiment = field_experiment[::4, ::4]
#     fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=1.1 * 6.5, stepsNumber=30, n0=1, k0=1)
#     fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 3, circle=True)
#     fOAM.plot_knot_dots(fieldAfterProp, axesAll=False, size=250, color='b')
#     plt.show()
#     exit()


# read_file_experiment_knot('3foil_noturb (5).mat')
_home = False
SR = '0.9 2'
knot = 'trefoil'
w = '1.6 (3)'  # Dima Cmex-

directoryName = f'C:\\WORK\\CODES\\knots_OAM_functions\\temp_data\Real\\Hopf_121313\\SR2 = 9.500000e-01\\'

table = directoryName.replace("\\", "_")[-25:-1]

tableName = f'.\\exels\\{table}'
# tableName = tableName.replace('\\', '_')

# directoryName = (f'C:\\Users\\Cmex-\Box\\Knots Exp\\New_Data\\'
#                  f'SR = {SR} (new)\\{knot}\\w = {w}/')
# directoryName = (
#     f'C:\\WORK\\CODES\\knots_OAM_functions'
#     f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
# if _home:
#     directoryName.replace('C:\\WORK\\CODES\\', 'C:\\SCIENCE\\programming\\Python\\gitHub\\')

kc.creat_knot_table(directoryName, tableName, single=None, show=True, cut=0.35)
exit()
