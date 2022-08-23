#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 15:36:22 2020

@author: albert, MCR

Utility functions for the edgetrigger algorithm
"""

import numpy as np
from scipy.optimize import least_squares


def zero_roll(a, shift):
    """Like np.roll but the wrapped around part is set to zero. Only works
    along the first axis of the array.

    Parameters
    ----------
    a : array
        The input array
    shift : int
        The number of rows to shift by.

    Returns
    -------
    result : np.array
        The input array with rows shifted.
    """

    result = np.zeros_like(a)
    if shift >= 0:
        result[shift:] = a[:-shift]
    else:
        result[:shift] = a[-shift:]

    return result


def robust_polyfit(x, y, order, maxiter=5, nstd=3.):
    """Perform a robust polynomial fit.

    Paremeters
    ----------
    x : array, list
        Independent fitting variable.
    y : array, list
        Dependent fitting variable.
    order : int
        Polynomial order to fit.
    maxiter : int
        Number of iterations for outlier rejection.
    nstd : float
        Number of standard deviations to consider a value an outlier.

    Returns
    -------
    res : array[float]
        The best fitting polynomial parameters.
    """

    def _poly_res(p, x, y):
        """Residuals from a polynomial.
        """
        return np.polyval(p, x) - y

    mask = np.ones_like(x, dtype='bool')
    for niter in range(maxiter):

        # Fit the data and evaluate the best-fit model.
        param = np.polyfit(x[mask], y[mask], order)
        yfit = np.polyval(param, x)

        # Compute residuals and mask ouliers.
        res = y - yfit
        stddev = np.std(res)
        mask = np.abs(res) <= nstd*stddev

    res = least_squares(_poly_res, param, loss='huber', f_scale=0.1,
                        args=(x[mask], y[mask])).x

    return res


def get_image_dim(image, header=None, verbose=False):
    """Determine the properties of the image array.

    Parameters
    ----------
    image : array[float]
        2D image of the detector.
    header : astropy.io.fits.Header instance
        The header from a SOSS reference file.
    verbose : bool
        If True, show diagnostic plots.

    Returns
    -------
    dimx : int
        X-dimension of the stack array.
    dimy : int
        Y-dimension of the stack array.
    xos : int
        Oversampling factor in the x-direction.
    yos : int
        Oversampling factor in the y-direction.
    xnative : int
        X-dimension of the stack array in native pixels.
    ynative : int
        Y-dimension of the stack array in native pixels.
    padding : int
        Amount of padding around the image, in native pixels.
    refpix_mask : array[bool]
        Boolean array indicating which pixels are light sensitive (True) and
        which are reference pixels (False).
    """

    # Dimensions of the subarray.
    dimy, dimx = np.shape(image)

    # If no header was passed we have to check all possible sizes.
    if header is None:

        # Initialize padding to zero in this case because it is not a
        # reference file.
        padding = 0

        # Assume the stack is a valid SOSS subarray.
        # FULL: 2048x2048 or 2040x2040 (working pixels) or multiple if os.
        # SUBSTRIP96: 2048x96 or 2040x96 (working pixels) or multiple if os.
        # SUBSTRIP256: 2048x256 or 2040x252 (working pixels) or multiple if os.
        # Check if the size of the x-axis is valid.
        if (dimx % 2048) == 0:
            xnative = 2048
            xos = int(dimx // 2048)
        elif (dimx % 2040) == 0:
            xnative = 2040
            xos = int(dimx // 2040)
        else:
            msg = ('Stack X dimension has unrecognized size of {:}. '
                   'Accepts 2048, 2040 or multiple thereof.')
            raise ValueError(msg.format(dimx))

        # Check if the y-axis is consistent with the x-axis.
        if np.int(dimy/xos) in [96, 256, 252, 2040, 2048]:
            yos = np.copy(xos)
            ynative = np.int(dimy/yos)
        else:
            msg = ('Stack Y dimension ({:}) is inconsistent with '
                   'stack X dimension ({:}) for acceptable SOSS arrays')
            raise ValueError(msg.format(dimy, dimx))

        # Create a boolean mask indicating which pixels are not ref pixels.
        refpix_mask = np.ones_like(image, dtype='bool')
        if xnative == 2048:
            # Mask out the left and right columns of reference pixels.
            refpix_mask[:, :xos * 4] = False
            refpix_mask[:, -xos * 4:] = False

        if ynative == 2048:
            # Mask out the top and bottom rows of reference pixels.
            refpix_mask[:yos * 4, :] = False
            refpix_mask[-yos * 4:, :] = False

        if ynative == 256:
            # Mask the top rows of reference pixels.
            refpix_mask[-yos * 4:, :] = False

    else:
        # Read the oversampling and padding from the header.
        padding = int(header['PADDING'])
        xos, yos = int(header['OVERSAMP']), int(header['OVERSAMP'])

        # Check that the stack respects its intended format.
        if (dimx/xos - 2*padding) not in [2048]:
            msg = 'The header passed is inconsistent with the X dimension ' \
                  'of the stack.'
            raise ValueError(msg)
        else:
            xnative = 2048

        if (dimy/yos - 2*padding) not in [96, 256, 2048]:
            msg = 'The header passed is inconsistent with the Y dimension ' \
                  'of the stack.'
            raise ValueError(msg)
        else:
            ynative = np.int(dimy/yos - 2*padding)

        # The trace file contains no reference pixels so all pixels are good.
        refpix_mask = np.ones_like(image, dtype='bool')

    # If verbose print the output.
    if verbose:
        print('Data dimensions:')
        str_args = dimx, dimy, xos, yos, xnative, ynative
        msg = 'dimx={:}, dimy={:}, xos={:}, yos={:}, xnative={:}, ynative={:}'
        print(msg.format(*str_args))

    return dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask
