# some standard python imports #
import sys
from termios import IEXTEN

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

import os
import astropy.io.fits as pyfits  # open / write FITS files
from PIL import Image  # images manipulation
import math
import pickle

from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.Util.param_util import shear_polar2cartesian
from lenstronomy.Util.param_util import shear_cartesian2polar
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData


from config import mask_sparsefit, kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model, kwargs_likelihood, kwargs_data_joint

#phi, gamma = -0.006, -70

phi, gamma = 0.4, 0.02

gamma1, gamma2 = shear_polar2cartesian(phi, gamma)
print(gamma1, gamma2)

results = {'kwargs_lens': [{'theta_E': 2.5735811407227294, 'e1': 0.13690516328659041, 'e2': -0.17317998080401018, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.8369583391719762, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.42208029731837404, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.24399471053752633, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 1, 'R_sersic': 0.49415932403070434, 'n_sersic': 2.1999503823654005, 'e1': -0.2681558460539695, 'e2': 0.08144060894415316, 'center_x': -0.4496148508821758, 'center_y': 0.5171749360342417}, {'amp': 1, 'n_max': 0, 'beta': 0.06428207523899651, 'center_x': -0.4496148508821758, 'center_y': 0.5171749360342417}], 'kwargs_lens_light': [{'amp': 1, 'R_sersic': 2.847141361585262, 'n_sersic': 3.9999713626322513, 'e1': 0.12559309716251224, 'e2': -0.012405656251595681, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.050000907388408476, 'n_sersic': 0.2969074517485267, 'e1': 0.18919008763124529, 'e2': -0.010156065351116746, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.40004009125230894, 'n_sersic': 3.9328508343188577, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 1, 'R_sersic': 0.11840998754165728, 'n_sersic': 0.9059417343861442, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 1, 'R_sersic': 0.3, 'n_sersic': 3.99, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}
resultsb = {'kwargs_lens': [{'theta_E': 2.704189959027299, 'e1': 0.11694193931880321, 'e2': -0.17725262343894427, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.6351615764432211, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.3875240992164326, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.1835340350076582, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 1, 'R_sersic': 0.2800080959611928, 'n_sersic': 2.1999442426986575, 'e1': -0.3753452625710587, 'e2': -0.027669184666051047, 'center_x': -0.5321408184931168, 'center_y': 0.472377302162932}, {'amp': 1, 'n_max': 0, 'beta': 0.05000093031659909, 'center_x': -0.5321408184931168, 'center_y': 0.472377302162932}], 'kwargs_lens_light': [{'amp': 1, 'R_sersic': 2.9132398751804347, 'n_sersic': 3.99999768510049, 'e1': 0.12367215441748701, 'e2': -0.014632261719867003, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.05000174072698606, 'n_sersic': 0.8392170052550798, 'e1': -0.4997087888540353, 'e2': 0.032199788991674756, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.4000001917688602, 'n_sersic': 3.9739884451925085, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 1, 'R_sersic': 0.11927900083128602, 'n_sersic': 0.950209053986823, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 1, 'R_sersic': 0.3, 'n_sersic': 3.99, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}

results2 = {'kwargs_lens': [{'theta_E': 2.6081870557416647, 'e1': 0.10885724905976975, 'e2': -0.1974248728830271, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.5885368864301515, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.5235280760802414, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.2097594839858223, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 1.2270554695427596, 'R_sersic': 0.18888297586452327, 'n_sersic': 1.0874936077103057, 'e1': -0.38980681195461636, 'e2': 0.06599121748094698, 'center_x': -0.4763962550587054, 'center_y': 0.4497753479085645}, {'amp': np.array([2.88860365]), 'n_max': 0, 'beta': 0.0738192521201779, 'center_x': -0.4763962550587054, 'center_y': 0.4497753479085645}], 'kwargs_lens_light': [{'amp': 0.1924609519608061, 'R_sersic': 3.0179738353066625, 'n_sersic': 3.999820172164147, 'e1': 0.10908384199685729, 'e2': -0.011338477091116036, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 14.873197534312718, 'R_sersic': 0.05000118898271522, 'n_sersic': 1.9631722972000658, 'e1': -0.49990912003377197, 'e2': -0.15675689728675224, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.437397563469883, 'R_sersic': 0.400135163478974, 'n_sersic': 3.9994574944863346, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.592727253541421, 'R_sersic': 0.12102199576315627, 'n_sersic': 0.9283796495621135, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 0.7064801064657296, 'R_sersic': 0.24811145904109044, 'n_sersic': 3.986524912733891, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}
results2b = {'kwargs_lens': [{'theta_E': 2.9521086802824614, 'e1': 0.12053946141125212, 'e2': -0.15296988479749196, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.37461715086042874, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.3842347857531024, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.2114518702861373, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 0.6056502231233899, 'R_sersic': 0.2518999618483833, 'n_sersic': 1.4682326488118185, 'e1': -0.36537150384955597, 'e2': 0.10135443355503919, 'center_x': -0.48855412768335954, 'center_y': 0.4013888082623661}, {'amp': np.array([7.65439699]), 'n_max': 0, 'beta': 0.08678315783578176, 'center_x': -0.48855412768335954, 'center_y': 0.4013888082623661}], 'kwargs_lens_light': [{'amp': 0.1962780908218284, 'R_sersic': 2.965395086505835, 'n_sersic': 3.9999672128394796, 'e1': 0.11544602740271138, 'e2': -0.013980664362018202, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 17.932233947038092, 'R_sersic': 0.05001353204373917, 'n_sersic': 1.2413737111321888, 'e1': -0.490396881480666, 'e2': -0.0024127101981357355, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.43957319419310464, 'R_sersic': 0.400113710710797, 'n_sersic': 3.9998175462514367, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.541562867328964, 'R_sersic': 0.12004925116896152, 'n_sersic': 0.9805651333607085, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 7.851784206997873, 'R_sersic': 0.0553028415924915, 'n_sersic': 3.986057474822968, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}

results3 = {'kwargs_lens': [{'theta_E': 2.9533903657048906, 'e1': 0.11592663670533389, 'e2': -0.15672044969350074, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.38697408418021473, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.3637347626645789, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.21612875296441456, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 1.0679931322846834, 'R_sersic': 0.20167634833930217, 'n_sersic': 2.199125995266744, 'e1': -0.35630269469007697, 'e2': 0.027021051595934976, 'center_x': -0.5154040418670564, 'center_y': 0.44137401379395547}, {'amp': np.array([2.88860365]), 'n_max': 8, 'beta': 0.053793614968753956, 'center_x': -0.5154040418670564, 'center_y': 0.44137401379395547}], 'kwargs_lens_light': [{'amp': 0.19769855621226992, 'R_sersic': 2.9431133499977515, 'n_sersic': 3.9999981250048173, 'e1': 0.12048488395695632, 'e2': -0.011048067278789781, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 14.499850305877654, 'R_sersic': 0.050004868320368746, 'n_sersic': 2.0693664269092835, 'e1': -0.49994382563120743, 'e2': -0.16879398129611825, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.44365038350396935, 'R_sersic': 0.40000629764016743, 'n_sersic': 3.9878656584430114, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.623597314019563, 'R_sersic': 0.11952100261128117, 'n_sersic': 0.959047684172584, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 4.900265411998335, 'R_sersic': 0.07703828638249531, 'n_sersic': 3.9837817035290826, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}

results4 = {'kwargs_lens': [{'theta_E': 2.692547169638314, 'e1': 0.1492994792241008, 'e2': -0.0687635495031888, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.6156637455211599, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.4021764796859079, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.19789478695901072, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0.009925328992151441, 'gamma2': 0.03557385239019394, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 0.2148171111021246, 'R_sersic': 0.3729555643846012, 'n_sersic': 2.1897264737857105, 'e1': -0.4025453439450054, 'e2': 0.036206750178217485, 'center_x': -0.48224090310848206, 'center_y': 0.4695648497010366}, {'amp': np.array([11.09117461]), 'n_max': 0, 'beta': 0.08588667052622, 'center_x': -0.48224090310848206, 'center_y': 0.4695648497010366}], 'kwargs_lens_light': [{'amp': 0.2012995896281008, 'R_sersic': 2.894877513535034, 'n_sersic': 3.9953395463082297, 'e1': 0.1254901054112447, 'e2': -0.020104729425877876, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 16.196317542874624, 'R_sersic': 0.05000989375449468, 'n_sersic': 1.6654870112157232, 'e1': -0.49197107132898676, 'e2': -0.17251321335731026, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.4732099228732642, 'R_sersic': 0.4004569700130262, 'n_sersic': 3.8567363339875103, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.839130546373625, 'R_sersic': 0.1179793613751283, 'n_sersic': 0.9190748090047871, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 7.54696173006463, 'R_sersic': 0.0531688277930643, 'n_sersic': 3.98794751657628, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}

results5 = {'kwargs_lens': [{'theta_E': 2.692547169638314, 'gamma': 1.80, 'e1': 0.1492994792241008, 'e2': -0.0687635495031888, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.6156637455211599, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.4021764796859079, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.19789478695901072, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0.009925328992151441, 'gamma2': 0.03557385239019394, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 0.2148171111021246, 'R_sersic': 0.3729555643846012, 'n_sersic': 2.1897264737857105, 'e1': -0.4025453439450054, 'e2': 0.036206750178217485, 'center_x': -0.48224090310848206, 'center_y': 0.4695648497010366}, {'amp': np.array([11.09117461]), 'n_max': 0, 'beta': 0.08588667052622, 'center_x': -0.48224090310848206, 'center_y': 0.4695648497010366}], 'kwargs_lens_light': [{'amp': 0.2012995896281008, 'R_sersic': 2.894877513535034, 'n_sersic': 3.9953395463082297, 'e1': 0.1254901054112447, 'e2': -0.020104729425877876, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 16.196317542874624, 'R_sersic': 0.05000989375449468, 'n_sersic': 1.6654870112157232, 'e1': -0.49197107132898676, 'e2': -0.17251321335731026, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.4732099228732642, 'R_sersic': 0.4004569700130262, 'n_sersic': 3.8567363339875103, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.839130546373625, 'R_sersic': 0.1179793613751283, 'n_sersic': 0.9190748090047871, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 7.54696173006463, 'R_sersic': 0.0531688277930643, 'n_sersic': 3.98794751657628, 'center_x': -3.8, 'center_y': -1.41}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}

######   INITIAL   ########  

q0a = 0.9
phia = 0
e1a,e2a = phi_q2_ellipticity(phia, q0a)
e1b,e2b = 0,0
e1c,e2c = 0,0
shiftx, shifty = -0.5, 0.2 
xa, ya =  0.065, 0.070 #0.54 + shiftx + 0.04, -0.13 + shifty - 0.01
xb, yb =  -0.65, 1.289 #-0.38 + shiftx + 0.23, 0.98 + shifty + 0.1
xc, yc =  1.849, 0.333 #2.23 + shiftx + 0.1 , 0.33 + shifty - 0.2
xd, yd = -3.8, -1.41 #-3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_init = results5.get('kwargs_lens')
kwargs_source_init = results5.get('kwargs_source')
kwargs_lens_light_init = results5.get('kwargs_lens_light')


# kwargs_lens_init = [{'theta_E': 2.687,'e1': 0.131,'e2': -0.176,'center_x':xa,'center_y': ya},
#                     {'theta_E': 0.642,'e1': e1b,'e2': e2b,'center_x': xb,'center_y': yb},
#                     {'theta_E': 0.432,'e1': e1c,'e2':e2c,'center_x': xc,'center_y': yc},
#                     {'theta_E': 0.142,'center_x': xd,'center_y': yd},
#                     {'gamma1': 0.0, 'gamma2': 0.0, 'ra_0': 0.0,'dec_0': 0.0}
# ]

# phi_source, q0_source = 10, 0.8
# e1_source, e2_source = phi_q2_ellipticity(phi_source, q0_source)

# kwargs_source_init = [{'amp': 40, 'R_sersic': 0.166, 'n_sersic': 0.772, 'e1': -0.354, 'e2': 0.042, 
#                        'center_x': -0.456, 'center_y': 0.441},
#                       {'n_max': 0.0 ,'beta': 0.3,'center_x': -0.5, 'center_y': 0.46}
# ]

# kwargs_lens_light_init = [{'amp': 100,'R_sersic': 2.975,'n_sersic': 3.99,'e1': 0.11,'e2': -0.004,'center_x':xa,'center_y': ya},
#                           {'amp': 100,'R_sersic': 0.05,'n_sersic': 0.304,'e1': 0.15,'e2': -0.115,'center_x':xa,'center_y': ya},
#                           {'amp': 20,'R_sersic': 0.401,'n_sersic': 3.99,'e1': e1b,'e2':e2b,'center_x':xb,'center_y': yb},
#                           {'amp': 20,'R_sersic': 0.118,'n_sersic': 0.89,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc},
#                           {'amp': 20,'R_sersic': 0.1,'n_sersic': 3.99,'center_x':xd,'center_y': yd}
# ]


kwargs_initial = {'kwargs_lens': kwargs_lens_init,
                  'kwargs_source': kwargs_source_init,
                  'kwargs_lens_light': kwargs_lens_light_init,
}

lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_initial, arrow_size=0.02,
                    image_likelihood_mask_list = [mask_sparsefit])


f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
lensPlot.data_plot(ax=axes[0])
lensPlot.model_plot(ax=axes[1])
f.tight_layout()
f.savefig('initial1.jpg', dpi=150)
#plt.show()
plt.close()

f, axes = lensPlot.plot_main()
f.savefig('initial1a.jpg', dpi=150)
#plt.show()
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig('initial1b.jpg', dpi=150)
#plt.show()
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig('initial1c.jpg', dpi=150)
#plt.show()
plt.close()







######   Upper   ########  


q0a = 1.0

phia = 40

e1a,e2a = 0.5, 0.5
e1b,e2b = 0, 0
e1c,e2c = 0., 0.

shiftx, shifty = -0.5 + 0.02, 0.2 + 0.06

xa, ya = 0.54 + shiftx + 0.03, -0.13 + shifty - 0.01
xb, yb = -0.38 + shiftx + 0.23, 0.98 + shifty + 0.1
xc, yc = 2.23 + shiftx + 0.1 , 0.33 + shifty - 0.2
xd, yd = -3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_upper = [{'theta_E': 3., 'gamma': 2.4, 'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 1.5,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.7,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.4,'center_x':xd,'center_y': yd},
                    {'gamma1': 100, 'gamma2': 1, 'ra_0': 0.0,'dec_0': 0.0}
]

phi_source, q0_source = 50, 1.

e1_source, e2_source = 0.5, 0.5

kwargs_source_upper = [{'R_sersic': 0.5, 'n_sersic': 2.2, 'e1': e1_source, 'e2': e2_source, 
                       'center_x': -0.2, 'center_y': 0.7},
                       {'n_max': 0 ,'beta': 1,'center_x': -0.2, 'center_y': 0.7}
]

kwargs_lens_light_upper = [{'R_sersic': 5.,'n_sersic': 4,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 5.,'n_sersic': 4,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 1.,'n_sersic': 4,'e1': e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                          {'R_sersic': 0.4,'n_sersic': 2,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                          {'R_sersic': 0.5,'n_sersic': 4.0,'center_x': xd,'center_y': yd}
]













######   Lower  ########  


q0a = 0.5

phia = -40

e1a,e2a = -0.5, -0.5
e1b,e2b = 0, 0
e1c,e2c = 0, 0

shiftx, shifty = -0.5 - 0.02, 0.2 - 0.02

xa, ya = 0.54 + shiftx + 0.03 - 0.05, -0.13 + shifty - 0.01
xb, yb = -0.38 + shiftx + 0.23, 0.98 + shifty + 0.1
xc, yc = 2.23 + shiftx + 0.1 , 0.33 + shifty - 0.2
xd, yd = -3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_lower = [{'theta_E':1., 'gamma': 1.2, 'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 0.2,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.2,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.01,'center_x':xd,'center_y': yd},
                    {'gamma1': -100, 'gamma2': -1, 'ra_0': 0.0,'dec_0': 0.0}
]

phi_source, q0_source = -30, 0.5

e1_source, e2_source = -0.5, -0.5

kwargs_source_lower = [{'R_sersic': 0.05, 'n_sersic': 0.5, 'e1': e1_source, 'e2': e2_source, 
                       'center_x': -0.80,'center_y': 0.3},
                       {'n_max': 0 ,'beta': 0.02,'center_x': -0.8, 'center_y': 0.3}
]

kwargs_lens_light_lower = [{'R_sersic': 0.3,'n_sersic': 2,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 0.05,'n_sersic': 0.1,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 0.4,'n_sersic': 1.5,'e1': e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                          {'R_sersic': 0.05,'n_sersic': 0.5,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                          {'R_sersic': 0.05,'n_sersic': 3.98,'center_x':xd,'center_y': yd}
]











######   Sigma  ########  

q0a = 0.2
q0b = 0.2
q0c = 0.2

phia = 30
phib = 30
phic = 30

e1a,e2a = 0.1, 0.1
e1b,e2b = 0, 0
e1c,e2c = 0., 0.



xa, ya = 0.02, 0.02
xb, yb = 0.02, 0.02
xc, yc = 0.02, 0.02
xd, yd = 0.02, 0.02

kwargs_lens_sigma = [{'theta_E': 0.5, 'gamma': 1., 'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 0.5,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.5,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.2,'center_x':xd,'center_y': yd},
                    {'gamma1': 10, 'gamma2': 0.1, 'ra_0': 0.0,'dec_0': 0.0}
]

phi_source, q0_source = 30, 0.2

e1_source, e2_source = 0.2, 0.2

kwargs_source_sigma = [{'R_sersic': 0.1, 'n_sersic': 0.2, 'e1': e1_source, 'e2': e2_source, 
                       'center_x': 0.5,'center_y': 0.3},
                       {'n_max': 0 ,'beta': 0.5,'center_x': 0.5, 'center_y': 0.3}
]

kwargs_lens_light_sigma = [{'R_sersic': 0.3,'n_sersic': 1,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 0.3,'n_sersic': 1,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                          {'R_sersic': 0.2,'n_sersic': 1,'e1': e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                          {'R_sersic': 0.2,'n_sersic': 1,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                          {'R_sersic': 0.1,'n_sersic': 0.1,'center_x':xd,'center_y': yd}
]







xa, ya =  0.065, 0.070 #0.54 + shiftx + 0.04, -0.13 + shifty - 0.01
xb, yb =  -0.65, 1.289 #-0.38 + shiftx + 0.23, 0.98 + shifty + 0.1
xc, yc =  1.849, 0.333 #2.23 + shiftx + 0.1 , 0.33 + shifty - 0.2
xd, yd = -3.8, -1.41 #-3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_fixed = [{'center_x': xa, 'center_y': ya},{'e1': 0, 'e2': 0, 'center_x': xb, 'center_y': yb},{'e1': 0, 'e2': 0, 'center_x': xc, 'center_y': yc},{'center_x': xd, 'center_y': yd},{'ra_0': 0, 'dec_0': 0}]

kwargs_source_fixed = [{},{'n_max': 0}]

kwargs_lens_light_fixed = [{'center_x': xa, 'center_y': ya},{'center_x': xa, 'center_y': ya},{'e1': 0, 'e2': 0, 'center_x': xb, 'center_y': yb},{'e1': 0, 'e2': 0, 'center_x': xc, 'center_y': yc}, {'center_x': xd, 'center_y': yd}]


kwargs_constraints = {'joint_lens_with_light': [[0, 0, ['center_x', 'center_y']], [2, 1, ['center_x', 'center_y','e1','e2']], [3, 2, ['center_x', 'center_y','e1','e2']], [4, 3, ['center_x', 'center_y']]],
                    'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y']]],
                    'joint_source_with_source': [[0, 1, ['center_x', 'center_y']]]
}

lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lens_lower, kwargs_lens_upper]
source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_source_lower, kwargs_source_upper]
lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lens_light_lower, kwargs_lens_light_upper]

kwargs_params = {'lens_model': lens_params,
                'source_model': source_params,
                'lens_light_model': lens_light_params}


fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params, mpi=False)

param = fitting_seq.param_class
args = param.kwargs2args(kwargs_lens=kwargs_lens_init, kwargs_source=kwargs_source_init, kwargs_lens_light=kwargs_lens_light_init)  
llhood = fitting_seq.likelihoodModule.logL(args, verbose=True)

lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed, kwargs_lens_lower, kwargs_lens_upper]
source_params = [kwargs_source_init, kwargs_source_sigma, kwargs_source_fixed, kwargs_source_lower, kwargs_source_upper]
lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, kwargs_lens_light_fixed, kwargs_lens_light_lower, kwargs_lens_light_upper]
