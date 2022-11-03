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

fitting_kwargs_list = [['update_settings', {'source_add_fixed' : [[1, ['n_max'] ,[int(sys.argv[1])]]]}],
                       ['update_settings', {'lens_add_fixed' : [[0, ['gamma'], [float(sys.argv[2])]]]}],
                       ['PSO', {'sigma_scale': 1., 'n_particles': 300, 'n_iterations': 230, 'threadCount': 16}]
                      ]

chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result = fitting_seq.best_fit()

param = fitting_seq.param_class
argsnew = param.kwargs2args(kwargs_lens=kwargs_result.get('kwargs_lens'), kwargs_source=kwargs_result.get('kwargs_source'), kwargs_lens_light=kwargs_result.get('kwargs_lens_light'))  
llhood = fitting_seq.likelihoodModule.logL(argsnew, verbose=True)

bic = fitting_seq.bic

print(bic)

file = open(f'LLhood{sys.argv[3]}new.txt', 'w+')
result = str(llhood) 
file.write(result)
file.close()

file = open(f'bic{sys.argv[3]}new.txt', 'w+')
result = str(bic) 
file.write(result)
file.close()


for i in range(len(chain_list)):
    f, axes = chain_plot.plot_chain_list(chain_list, i)
    f.savefig(f'convergence{sys.argv[3]}new.jpg', dpi=150)
plt.close()
    
    
lensPlot = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])

file = open(f"kwargs{sys.argv[3]}new.txt", "w+")
result = str(kwargs_result)
file.write(result)
file.close()
    

f, axes = lensPlot.plot_main()
f.savefig(f'mainplot{sys.argv[3]}new.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_separate()
f.savefig(f'separate{sys.argv[3]}new.jpg', dpi=150)
plt.close()

f, axes = lensPlot.plot_subtract_from_data_all()
f.savefig(f'subtract{sys.argv[3]}new.jpg', dpi=150)
plt.close()














fitting_kwargs_list = [['update_settings', {'lens_remove_fixed' : [[0, ['gamma']]]}],
                       ['MCMC', {'n_burn': 0, 'n_run': 400, 'walkerRatio' : 10, 'sigma_scale': 0.2, 'threadCount': 16}] #'walkerRatio' : 10 et enlever n_waklers, nburn = 0 
                      ]

chain_list2 = fitting_seq.fit_sequence(fitting_kwargs_list)
kwargs_result2 = fitting_seq.best_fit()

print(np.array(chain_list2).shape)

param2 = fitting_seq.param_class
argsnew2 = param2.kwargs2args(kwargs_lens=kwargs_result2.get('kwargs_lens'), kwargs_source=kwargs_result2.get('kwargs_source'), kwargs_lens_light=kwargs_result2.get('kwargs_lens_light'))  
llhood2 = fitting_seq.likelihoodModule.logL(argsnew2, verbose=True)

bic2 = fitting_seq.bic

print(bic2)

file = open(f'LLhood{sys.argv[3]}new2.txt', 'w+')
result2 = str(llhood2) 
file.write(result2)
file.close()

file = open(f'bic{sys.argv[3]}new2.txt', 'w+')
result2 = str(bic2) 
file.write(result2)
file.close()


for i in range(len(chain_list2)):
    f, axes = chain_plot.plot_chain_list(chain_list2, i)
    f.savefig(f'convergence{sys.argv[3]}new2.jpg', dpi=150)
plt.close()
    
    
lensPlot2 = ModelPlot([[kwargs_data, kwargs_psf, kwargs_numerics]], kwargs_model, kwargs_result2, arrow_size=0.02, cmap_string="gist_heat",
                     image_likelihood_mask_list = [mask_sparsefit])

file = open(f"kwargs{sys.argv[3]}new2.txt", "w+")
result2 = str(kwargs_result2)
file.write(result2)
file.close()
    

f, axes = lensPlot2.plot_main()
f.savefig(f'mainplot{sys.argv[3]}new2.jpg', dpi=150)
plt.close()

f, axes = lensPlot2.plot_separate()
f.savefig(f'separate{sys.argv[3]}new2.jpg', dpi=150)
plt.close()

f, axes = lensPlot2.plot_subtract_from_data_all()
f.savefig(f'subtract{sys.argv[3]}new2.jpg', dpi=150)
plt.close()



from lenstronomy.Plots import chain_plot

param3 = fitting_seq.param_class
param_array_truths = param3.kwargs2args(kwargs_lens=kwargs_result2.get('kwargs_lens'), kwargs_source=kwargs_result2.get('kwargs_source'), kwargs_lens_light=kwargs_result2.get('kwargs_lens_light'))
    
sampler_type, samples_mcmc, param_mcmc, dist_mcmc  = chain_list2[0]
print("number of non-linear parameters in the MCMC process: ", len(param_mcmc))
print("parameters in order: ", param_mcmc)
print("number of evaluations in the MCMC process: ", np.shape(samples_mcmc)[0])

import corner  # pip install corner  (if you have not installed it)
import mcmc

import pickle
with open('chain_list{sys.argv[3]}.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(chain_list2[0], file)


n, num_param = np.shape(samples_mcmc)
plot = corner.corner(samples_mcmc[:,:8], labels=param_mcmc[:8], show_titles=True, truths=param_array_truths[:8])
plot.savefig(f'corner1{sys.argv[3]}.jpg',dpi=150)
plt.close()
plot = corner.corner(samples_mcmc[:,8:], labels=param_mcmc[8:], show_titles=True, truths=param_array_truths[8:])
plot.savefig(f'corner2{sys.argv[3]}.jpg',dpi=150)
plt.close()


fig = mcmc.plot_convergence_by_walker(samples_mcmc, param_mcmc, walkerRatio=10, n_iter_total=None, 
                               verbose=False, display_autocorr_time=True)
fig.savefig(f'mcmc{sys.argv[3]}.jpg', dpi=150)
plt.close()

