import config

# some standard python imports #
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

import os
import astropy.io.fits as pyfits  # open / write FITS files
from PIL import Image  # images manipulation
import math

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot

from initb import fitting_seq, mask_sparsefit, kwargs_data, kwargs_psf, kwargs_numerics, lens_params, source_params, lens_light_params, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood

fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 700, 'threadCount': 16}]
                      ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

param = fitting_seq.param_class
argsnew = param.kwargs2args(kwargs_lens=kwargs_result.get('kwargs_lens'), kwargs_source=kwargs_result.get('kwargs_source'), kwargs_lens_light=kwargs_result.get('kwargs_lens_light'))  
llhood = fitting_seq.likelihoodModule.logL(argsnew, verbose=True)

file = open('LLhood1b.txt', 'w+')
result = str(llhood)
file.write(result)
file.close()

for i in range(len(chain_list)):
    f, axes = chain_plot.plot_chain_list(chain_list, i)
    f.savefig('convergence1b.jpg', dpi=150)
plt.close()
    
    
lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])

file = open("kwargs1b.txt", "w+")
result = str(kwargs_result)
file.write(result)
file.close()
    

f, axes = lensPlot.plot_main()
f.savefig('mainplot1b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig('separate1b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig('subtract1b.jpg', dpi=150)
plt.close()





fitting_kwargs_list = [['update_settings', {'lens_remove_fixed' : [[0, ['gamma'] ]]}],
                       ['PSO', {'sigma_scale': 1., 'n_particles': 400, 'n_iterations': 1000, 'threadCount': 16}]
                      ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

param = fitting_seq.param_class
argsnew = param.kwargs2args(kwargs_lens=kwargs_result.get('kwargs_lens'), kwargs_source=kwargs_result.get('kwargs_source'), kwargs_lens_light=kwargs_result.get('kwargs_lens_light'))  
llhood = fitting_seq.likelihoodModule.logL(argsnew, verbose=True)

file = open('LLhood2b.txt', 'w+')
result = str(llhood)
file.write(result)
file.close()

for i in range(len(chain_list)):
    f, axes = chain_plot.plot_chain_list(chain_list, i)
    f.savefig('convergence2b.jpg', dpi=150)
plt.close()
    

lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])
    
    
file = open("kwargs2b.txt", "w+")
result = str(kwargs_result)
file.write(result)
file.close()

f, axes = lensPlot.plot_main()
f.savefig('mainplot2b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig('separate2b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig('subtract2b.jpg', dpi=150)
plt.close()


fitting_kwargs_list = [['update_settings', {'source_add_fixed' : [[1, ['n_max'] ,[6]]]}],
                       ['PSO', {'sigma_scale': 1., 'n_particles': 400, 'n_iterations': 1000, 'threadCount': 16}]
                      ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

param = fitting_seq.param_class
argsnew = param.kwargs2args(kwargs_lens=kwargs_result.get('kwargs_lens'), kwargs_source=kwargs_result.get('kwargs_source'), kwargs_lens_light=kwargs_result.get('kwargs_lens_light'))  
llhood = fitting_seq.likelihoodModule.logL(argsnew, verbose=True)

file = open('LLhood3b.txt', 'w+')
result = str(llhood)
file.write(result)
file.close()

for i in range(len(chain_list)):
    f, axes = chain_plot.plot_chain_list(chain_list, i)
    f.savefig('convergence3b.jpg', dpi=150)
plt.close()
    

lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])
    
    
file = open("kwargs3b.txt", "w+")
result = str(kwargs_result)
file.write(result)
file.close()

f, axes = lensPlot.plot_main()
f.savefig('mainplot3b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig('separate3b.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig('subtract3b.jpg', dpi=150)
plt.close()