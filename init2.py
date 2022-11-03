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
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.imaging_data import ImageData


from config2 import mask_sparsefit, kwargs_data, kwargs_psf, kwargs_numerics, kwargs_model, kwargs_likelihood, kwargs_data_joint


#results = {'kwargs_lens': [{'theta_E': 2.5735811407227294, 'e1': 0.13690516328659041, 'e2': -0.17317998080401018, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.8369583391719762, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.42208029731837404, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.024399471053752633, 'center_x': -3.64, 'center_y': -1.68}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 1, 'R_sersic': 0.49415932403070434, 'n_sersic': 2.1999503823654005, 'e1': -0.2681558460539695, 'e2': 0.08144060894415316, 'center_x': -0.4496148508821758, 'center_y': 0.5171749360342417}, {'amp': 1, 'n_max': 10, 'beta': 0.06428207523899651, 'center_x': -0.4496148508821758, 'center_y': 0.5171749360342417}], 'kwargs_lens_light': [{'amp': 1, 'R_sersic': 2.847141361585262, 'n_sersic': 3.9999713626322513, 'e1': 0.12559309716251224, 'e2': -0.012405656251595681, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.050000907388408476, 'n_sersic': 0.2969074517485267, 'e1': 0.18919008763124529, 'e2': -0.010156065351116746, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 1, 'R_sersic': 0.40004009125230894, 'n_sersic': 3.9328508343188577, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 1, 'R_sersic': 0.11840998754165728, 'n_sersic': 0.9059417343861442, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'amp': 1, 'R_sersic': 0.3, 'n_sersic': 3.99, 'center_x': -3.64, 'center_y': -1.68}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}
results = {'kwargs_lens': [{'theta_E': 2.9521086802824614, 'e1': 0.12053946141125212, 'e2': -0.15296988479749196, 'center_x': 0.065, 'center_y': 0.07}, {'theta_E': 0.37461715086042874, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'theta_E': 0.3842347857531024, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}, {'theta_E': 0.2114518702861373, 'center_x': -3.8, 'center_y': -1.41}, {'gamma1': 0, 'gamma2': 0, 'ra_0': 0, 'dec_0': 0}], 'kwargs_source': [{'amp': 0.6056502231233899, 'R_sersic': 0.2518999618483833, 'n_sersic': 1.4682326488118185, 'e1': -0.36537150384955597, 'e2': 0.10135443355503919, 'center_x': -0.48855412768335954, 'center_y': 0.4013888082623661}, {'amp': np.array([7.65439699]), 'n_max': 0, 'beta': 0.08678315783578176, 'center_x': -0.48855412768335954, 'center_y': 0.4013888082623661}], 'kwargs_lens_light': [{'amp': 0.1962780908218284, 'R_sersic': 2.965395086505835, 'n_sersic': 3.9999672128394796, 'e1': 0.11544602740271138, 'e2': -0.013980664362018202, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 17.932233947038092, 'R_sersic': 0.05001353204373917, 'n_sersic': 1.2413737111321888, 'e1': -0.490396881480666, 'e2': -0.0024127101981357355, 'center_x': 0.065, 'center_y': 0.07}, {'amp': 0.43957319419310464, 'R_sersic': 0.400113710710797, 'n_sersic': 3.9998175462514367, 'e1': 0, 'e2': 0, 'center_x': -0.65, 'center_y': 1.289}, {'amp': 6.541562867328964, 'R_sersic': 0.12004925116896152, 'n_sersic': 0.9805651333607085, 'e1': 0, 'e2': 0, 'center_x': 1.849, 'center_y': 0.333}], 'kwargs_ps': [], 'kwargs_special': {}, 'kwargs_extinction': []}


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
xd, yd =  -3.8, -1.41 #-3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_init = results.get('kwargs_lens')
kwargs_source_init = results.get('kwargs_source')
kwargs_lens_light_init = results.get('kwargs_lens_light')


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
#                       {'n_max': 0.0 ,'beta': 0.2,'center_x': -0.5, 'center_y': 0.46}
# ]

# kwargs_lens_light_init = [{'amp': 100,'R_sersic': 2.975,'n_sersic': 3.99,'e1': 0.11,'e2': -0.004,'center_x':xa,'center_y': ya},
#                           {'amp': 100,'R_sersic': 0.05,'n_sersic': 0.304,'e1': 0.15,'e2': -0.115,'center_x':xa,'center_y': ya},
#                           {'amp': 20,'R_sersic': 0.401,'n_sersic': 3.99,'e1': e1b,'e2':e2b,'center_x':xb,'center_y': yb},
#                           {'amp': 20,'R_sersic': 0.118,'n_sersic': 0.89,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc},
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
f.savefig('initialtest.jpg', dpi=150)
#plt.show()
plt.close()

f, axes = lensPlot.plot_main()
f.savefig('initialtesta.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig('initialtestb.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig('initialtestc.jpg', dpi=150)
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

kwargs_lens_upper = [{'theta_E': 3.,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 1.5,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.7,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.4,'center_x':xd,'center_y': yd},
                    {'gamma1': 0, 'gamma2': 0, 'ra_0': 0.0,'dec_0': 0.0}
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
                          {'R_sersic': 0.4,'n_sersic': 2,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc}
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

kwargs_lens_lower = [{'theta_E':1.,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 0.2,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.2,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.01,'center_x':xd,'center_y': yd},
                    {'gamma1': 0, 'gamma2': 0, 'ra_0': 0.0,'dec_0': 0.0}
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
                          {'R_sersic': 0.05,'n_sersic': 0.5,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc}
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

kwargs_lens_sigma = [{'theta_E': 0.5,'e1': e1a,'e2':e2a,'center_x':xa,'center_y': ya},
                    {'theta_E': 0.5,'e1':e1b,'e2':e2b,'center_x':xb,'center_y': yb},
                    {'theta_E': 0.5,'e1':e1c,'e2':e2c,'center_x':xc,'center_y': yc},
                    {'theta_E': 0.2,'center_x':xd,'center_y': yd},
                    {'gamma1': 0.0, 'gamma2': 0.0, 'ra_0': 0.0,'dec_0': 0.0}
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
                          {'R_sersic': 0.2,'n_sersic': 1,'e1': e1c,'e2':e2c,'center_x':xc,'center_y': yc}
]



xa, ya =  0.065, 0.070 #0.54 + shiftx + 0.04, -0.13 + shifty - 0.01
xb, yb =  -0.65, 1.289 #-0.38 + shiftx + 0.23, 0.98 + shifty + 0.1
xc, yc =  1.849, 0.333 #2.23 + shiftx + 0.1 , 0.33 + shifty - 0.2
xd, yd =  -3.8, -1.41 #-3.12 + shiftx, -2.1 + shifty + 0.2

kwargs_lens_fixed = [{'center_x': xa, 'center_y': ya},{'e1': 0, 'e2': 0, 'center_x': xb, 'center_y': yb},{'e1': 0, 'e2': 0, 'center_x': xc, 'center_y': yc},{'center_x': xd, 'center_y': yd},{'gamma1': 0, 'gamma2': 0,'ra_0': 0, 'dec_0': 0}]

kwargs_source_fixed = [{},{'n_max': 0}]

kwargs_lens_light_fixed = [{'center_x': xa, 'center_y': ya},{'center_x': xa, 'center_y': ya},{'e1': 0, 'e2': 0, 'center_x': xb, 'center_y': yb},{'e1': 0, 'e2': 0, 'center_x': xc, 'center_y': yc}]


kwargs_constraints = {'joint_lens_with_light': [[0, 0, ['center_x', 'center_y']], [2, 1, ['center_x', 'center_y','e1','e2']], [3, 2, ['center_x', 'center_y','e1','e2']]],
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
