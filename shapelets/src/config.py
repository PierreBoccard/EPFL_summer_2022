# some standard python imports #
import sys

import numpy as np
import copy
import matplotlib.pyplot as plt

import os
import astropy
import astropy.cosmology
import astropy.io.fits as pyfits  # open / write FITS files
from PIL import Image  # images manipulation
import math
import matplotlib
matplotlib.use('Agg') 

from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Sampling.parameters import Param

from lenstronomy.Plots.model_plot import ModelPlot
from imd.image_mask import ImageMask
from lenstronomy.Util import class_creator 

    
lentille_raw = "/home/astro/pboccard/summerstage/data/hst_wfpc2_J1206_drz.fits"
data_lentillebis = pyfits.open(lentille_raw)
data_lentille = data_lentillebis[0].data[2020:2527,520:1027]

patch_bkg = data_lentillebis[0].data[2470:2620,220:370]
mediane = np.median(patch_bkg)
sigma_bkg = 1.48 * np.median(np.abs(patch_bkg - mediane))

data_lentille -= mediane

exposition_raw = "/home/astro/pboccard/summerstage/data/hst_wfpc2_J1206_wht.fits"
data_exposition = pyfits.open(exposition_raw)
data_exposition = data_exposition[0].data[2020:2527,520:1027]

delta = 26

psf_raw = "/home/astro/pboccard/summerstage/data/tinytim_wfpc2_f606w_J1206.fits"
data_psf = pyfits.open(psf_raw)
data_psf = data_psf[0].data[60 - delta : 61 + delta, 60 - delta : 61 + delta]

# header entries for the linear transformation (from a HST data header)
header = {'CD1_1': -1.3783350763348E-05,
            'CD1_2':  1.70894011082861E-06,
            'CD2_1': 1.70894011082861E-06,
            'CD2_2': 1.37833507633486E-05,
            'NAXIS1': 507,
            'NAXIS2': 507
            }

# read out matrix elements and convert them in units of arc seconds
CD1_1 = header.get('CD1_1') * 3600  # change in arc sec per pixel d(ra)/dx
CD1_2 = header.get('CD1_2') * 3600
CD2_1 = header.get('CD2_1') * 3600
CD2_2 = header.get('CD2_2') * 3600

# generate pixel-to-coordinate transform matrix and its inverse
pix2coord_transform_undistorted = np.array([[CD1_1, CD1_2], [CD2_1, CD2_2]])
det = CD1_1*CD2_2 - CD1_2*CD2_1
coord2pix_transform_undistorted = np.array([[CD2_2, -CD1_2], [-CD2_1, CD1_1]])/det

# as an example, we set the coordinate zero point in the center of the image and compute
# the coordinate at the pixel (0,0) at the edge of the image

# read out pixel size of image
nx = header.get('NAXIS1')
ny = header.get('NAXIS2')
x_c = int(nx / 2)
y_c = int(ny / 2)

# compute RA/DEC relative shift between the edge and the center of the image
dra, ddec = pix2coord_transform_undistorted.dot(np.array([x_c, y_c]))
# set edge of the image such that the center has RA/DEC = (0,0)
ra_at_xy_0, dec_at_xy_0 = -dra, -ddec

kwargs_data = {
    'image_data': data_lentille,
    'exposure_time': data_exposition,
    'background_rms': sigma_bkg,
    'ra_at_xy_0': ra_at_xy_0,
    'dec_at_xy_0': dec_at_xy_0,
    'transform_pix2angle': pix2coord_transform_undistorted,
}

kwargs_psf = {
    'psf_type': 'PIXEL',
    'kernel_point_source': data_psf,
}

kwargs_data_sub = copy.deepcopy(kwargs_data)
data_sub_class = ImageData(**kwargs_data_sub)

centers = [(12.7,12.2), (6.2, 15), (13.5, 4.25), (11.5,5.85), (8.5,16.35), (14.4, 17.35), (14.25,20.25), (16.75,6.25), (19.65,11.35), (14.5,7.9), (15.7,18.7)]
radius = [8, 0.65, 0.4, 0.5, 0.5, 0.15, 0.4, 1.2, 0.25, 0.3, 0.25]
phi = [90, 0, -20, 90, 90, 0, 0, 30, 1, 1, 135]
q0 = [1.11, 1, 3, 1.66, 1.05, 1, 8, 0.33, 1, 1, 2.8]
inverted = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
operation = ['inter','inter','inter','inter','inter','inter','inter','inter','inter','inter']

num_pix = data_sub_class.data.shape
delta_pix = data_sub_class.pixel_width

mask_kwargs = {
            'mask_type': 'ellipse',
            'radius_list': radius,
            'center_list': centers,
            'angle_list': phi,
            'axis_ratio_list': q0,
            'inverted_list': inverted,
            'operation_list': operation,
    }
imgMask = ImageMask(mask_shape=data_sub_class.data.shape,
                    delta_pix=data_sub_class.pixel_width,
                    **mask_kwargs)

mask_sparsefit = imgMask.get_mask(show_details=False)

# specify the choice of lens models #
lens_model_list = ['EPL','SIE','SIE','SIS','SHEAR']
# setup lens model class with the list of lens models #

kwargs_lens = {'lens_model_list': lens_model_list,
}

# set up the list of light models to be used #
source_light_model_list = ['SERSIC_ELLIPSE','SHAPELETS']
lens_light_model_list = ['SERSIC_ELLIPSE','SERSIC_ELLIPSE','SERSIC_ELLIPSE','SERSIC_ELLIPSE','SERSIC']

kwargs_light_source = { 'light_model_list':source_light_model_list
}

kwargs_lens_light = {'light_model_list': lens_light_model_list,
}

# define the numerics #
kwargs_numerics = {'supersampling_factor': 1, # each pixel gets super-sampled (in each axis direction)
                    'supersampling_convolution': False}

kwargs_model = {'lens_model_list': lens_model_list, 'source_light_model_list': source_light_model_list,
                'lens_light_model_list': lens_light_model_list} #, 'point_source_model_list': point_source_model_list}

kwargs_likelihood = {'image_likelihood': True,
                        'check_bounds': True,
                        'check_positive_flux': True,
                        'image_likelihood_mask_list': [mask_sparsefit],
                        #prior_lens= list of [index_model, param_name, mean, 1-sigma priors]
}
single_band = [[kwargs_data, kwargs_psf, kwargs_numerics]]
kwargs_data_joint = {'multi_band_list': single_band, 'multi_band_type': 'single-band'}

H0, OmM, OmL = 73.0, 0.25, 0.75
zlens, zsource = 0.422, 2.001 
c = 2.998e+5

def dx12(z, c, H0, omeg_M, omeg_lamb):
     return (c/H0/(np.sqrt((1-omeg_M-omeg_lamb)*(1+z)**2 + omeg_M*(1+z)**3 + omeg_lamb)))

def dist(x, z):
     return (x/(1+z))

xTL = dx12(zlens, c, H0, OmM, OmL)
distTL = dist(xTL, zlens)

xTS = dx12(zsource, c, H0, OmM, OmL)
distTS = dist(xTS, zsource)

from scipy.integrate import quad

x_lens = quad(dx12, 0, zlens, args=(c,H0,OmM, OmL))[0]
lens_dist = dist(x_lens,zlens)

x_source = quad(dx12, 0, zsource, args=(c,H0,OmM, OmL))[0]
source_dist = dist(x_source,zsource)

x_source_lens = quad(dx12, zlens, zsource, args=(c,H0,OmM, OmL))[0]
source_lens_dist = dist(x_source_lens,zsource)

print(lens_dist)
print(source_dist)
print(source_lens_dist)