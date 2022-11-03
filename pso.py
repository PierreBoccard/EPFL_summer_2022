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

from init import fitting_seq, mask_sparsefit, kwargs_data, kwargs_psf, kwargs_numerics, lens_params, source_params, lens_light_params, kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood

file = open(f'TEST{sys.argv[3]}.txt', 'w+')
result = sys.argv[2]
file.write(result)
file.close()

fitting_kwargs_list = [['update_settings', {'source_add_fixed' : [[1, ['n_max'] ,[int(sys.argv[1])]]]}],
                       ['update_settings', {'lens_add_fixed' : [[0, ['gamma'], [float(sys.argv[2])]]]}],
                       ['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 500, 'threadCount': 16}]
                      ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

param = fitting_seq.param_class
argsnew = param.kwargs2args(kwargs_lens=kwargs_result.get('kwargs_lens'), kwargs_source=kwargs_result.get('kwargs_source'), kwargs_lens_light=kwargs_result.get('kwargs_lens_light'))  
llhood = fitting_seq.likelihoodModule.logL(argsnew, verbose=True)

bic = fitting_seq.bic

print(bic)

file = open(f'LLhood{sys.argv[3]}bis.txt', 'w+')
result = str(llhood) 
file.write(result)
file.close()

file = open(f'bic{sys.argv[3]}bis.txt', 'w+')
result = str(bic) 
file.write(result)
file.close()


for i in range(len(chain_list)):
    f, axes = chain_plot.plot_chain_list(chain_list, i)
    f.savefig(f'convergence{sys.argv[3]}bis.jpg', dpi=150)
plt.close()
    
    
lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])

file = open(f"kwargs{sys.argv[3]}bis.txt", "w+")
result = str(kwargs_result)
file.write(result)
file.close()
    

f, axes = lensPlot.plot_main()
f.savefig(f'mainplot{sys.argv[3]}bis.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig(f'separate{sys.argv[3]}bis.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig(f'subtract{sys.argv[3]}bis.jpg', dpi=150)
plt.close()

