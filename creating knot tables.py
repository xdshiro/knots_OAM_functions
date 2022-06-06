import knot_class as kc
import my_modules.functions_general as fg
import my_modules.functions_OAM_knots as fOAM
import numpy as np
import matplotlib.pyplot as plt
def read_file_experiment_knot(fileName):
    # fileName = '3foil_field2_modIntensity_f.mat'
    field_experiment = fg.readingFile(fileName=fileName, fieldToRead='Uz',
                                      printV=False)
    fg.plot_2D(np.abs(field_experiment) ** 2, axis_equal=True)
    fg.plot_2D(np.angle(field_experiment), axis_equal=True)
    plt.show()
    # exit()
    # field_experiment = field_experiment[::4, ::4]
    fieldAfterProp = fg.one_plane_propagator(field_experiment, dz=1.1 * 6.5, stepsNumber=30, n0=1, k0=1)
    fieldAfterProp = fg.cut_filter(fieldAfterProp, radiusPix=np.shape(fieldAfterProp)[0] // 3.5, circle=True)
    fOAM.plot_knot_dots(fieldAfterProp, axesAll=False, size=250, color='b')
    plt.show()
    exit()

# read_file_experiment_knot('3foil_noturb (5).mat')
_home = False
SR = '0.95'
knot = '3foil'
w = 'stan'  # Dima Cmex-
folderInside = 'w = 1.5'
directoryName = (
    f'C:\\WORK\\CODES\\knots_OAM_functions'
    f'\\temp_data\\SR = 0.95 (opt w)\\{folderInside}\\')

tableName = f'{knot}, SR={SR}, w={w}, {folderInside}'
tableName.replace('\\', '_')
# directoryName = (f'C:\\Users\\Cmex-\Box\\Knots Exp\\New_Data\\'
#                  f'SR = {SR} (new)\\{knot}\\w = {w}/')
# directoryName = (
#     f'C:\\WORK\\CODES\\knots_OAM_functions'
#     f'\\temp_data\\SR = {SR}\\{knot}\\w = {w}\\')
if _home:
    directoryName.replace('C:\\WORK\\CODES\\', 'C:\\SCIENCE\\programming\\Python\\gitHub\\')

kc.creat_knot_table(directoryName, tableName, single=None, show=True, cut=0.35)
exit()
directoryName =f'.\\experiment\\trefoil\\SR = 0.9\\'
kc.creat_knot_table_dict_dots(directoryName, tableName)
exit()