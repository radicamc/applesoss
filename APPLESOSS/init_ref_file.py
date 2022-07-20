#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 15:35:32 2020

@author: albert

Routine to initialize a specprofile reference file.
"""

from astropy.io import fits
from datetime import datetime
import numpy as np
import os


class RefTraceTable:

    def __init__(self, filenames=None):

        if filenames is None:
            filenames = {'FULL': 'SOSS_ref_trace_table_FULL.fits.gz',
                         'SUBSTRIP96': 'SOSS_ref_trace_table_SUBSTRIP96.fits.gz',
                         'SUBSTRIP256': 'SOSS_ref_trace_table_SUBSTRIP256.fits.gz'}

        self.filenames = filenames

        return

    def __call__(self, column, wavelengths=None, subarray='SUBSTRIP256', order=1):

        if subarray not in ['FULL', 'SUBSTRIP96', 'SUBSTRIP256']:
            raise ValueError('Unknown subarray: {}'.format(subarray))

        filename = os.path.join(PATH, self.filenames[subarray])
        trace_table, header, = fits.getdata(filename, header=True, extname='ORDER', extver=order)

        if wavelengths is None:
            wavelengths = trace_table['WAVELENGTH']
            values = trace_table[column]
        else:
            values = np.interp(wavelengths, trace_table['WAVELENGTH'], trace_table[column], left=np.nan, right=np.nan)

        return wavelengths, values


def init_spec_profile(profile_2d, oversample, padding, subarray,
                      filename=None):
    """Initialize a specprofile reference file in the format expected by the
    JWST DMS.

    Parameters
    ----------
    profile_2d : array-like
        2D spatial profiles for each order.
    oversample : int
        Oversampling factor.
    padding : int
        Amount of padding.
    subarray : str
        SOSS subarray identifier.
    filename : str
        Output file name.

    Returns
    -------
    hdul : fits HDUList object
        Reference file in correct format.
    """

    # Default filename.
    if filename is None:
        # Output SOSS reference file.
        filepattern = 'SOSS_ref_2D_profile_{}.fits'
        filename = filepattern.format(subarray)

    # Find the indices in the FULL subarray for the requested subarrays.
    if subarray == 'FULL':
        lrow = 0
        urow = oversample * (2048 + 2 * padding)
        lcol = 0
        ucol = oversample * (2048 + 2 * padding)
    elif subarray == 'SUBSTRIP96':
        lrow = oversample * (2048 - 246)
        urow = oversample * (2048 - 150 + 2 * padding)
        lcol = 0
        ucol = oversample * (2048 + 2 * padding)
    elif subarray == 'SUBSTRIP256':
        lrow = oversample * (2048 - 256)
        urow = oversample * (2048 + 2 * padding)
        lcol = 0
        ucol = oversample * (2048 + 2 * padding)
    else:
        raise ValueError('Unknown subarray: {}'.format(subarray))

    # Start building the output fits file.
    hdul = list()
    hdu = fits.PrimaryHDU()
    hdu.header['DATE'] = (datetime.now().strftime('%Y-%m-%dT%H:%M:%S'), 'Date this file was created (UTC)')
    hdu.header['ORIGIN'] = ('APPLESOSS', 'Orginazation responsible for creating file')
    hdu.header['TELESCOP'] = ('JWST', 'Telescope used to acquire the data')
    hdu.header['INSTRUME'] = ('NIRISS', 'Instrument used to acquire the data')
    hdu.header['SUBARRAY'] = (subarray, 'Subarray used')
    hdu.header['FILENAME'] = (filename, 'Name of the file')
    hdu.header['REFTYPE'] = ('SPECPROFILE', 'Reference file type')
    hdu.header['PEDIGREE'] = ('GROUND', 'The pedigree of the refernce file')
    hdu.header['DESCRIP'] = ('2D trace profile', 'Desription of the reference file')
    hdu.header['AUTHOR'] = ('Loic Albert', 'Author of the reference file')
    hdu.header['USEAFTER'] = ('2000-01-01T00:00:00', 'Use after date of the reference file')
    hdu.header['EXP_TYPE'] = ('NIS_SOSS', 'Type of data in the exposure')
    hdul.append(hdu)

    # The order 1 profile.
    hdu = fits.ImageHDU(profile_2d[lrow:urow, lcol:ucol, 0].astype('float32'))
    hdu.header['ORDER'] = (1, 'Spectral order.')
    hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')
    hdu.header['EXTNAME'] = 'ORDER'
    hdu.header['EXTVER'] = 1
    hdul.append(hdu)

    # The order 2 profile.
    hdu = fits.ImageHDU(profile_2d[lrow:urow, lcol:ucol, 1].astype('float32'))
    hdu.header['ORDER'] = (2, 'Spectral order.')
    hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')
    hdu.header['EXTNAME'] = 'ORDER'
    hdu.header['EXTVER'] = 2
    hdul.append(hdu)

    # The order 3 profile.
    hdu = fits.ImageHDU(profile_2d[lrow:urow, lcol:ucol, 2].astype('float32'))
    hdu.header['ORDER'] = (3, 'Spectral order.')
    hdu.header['OVERSAMP'] = (oversample, 'Pixel oversampling.')
    hdu.header['PADDING'] = (padding, 'Native pixel-size padding around the image.')
    hdu.header['EXTNAME'] = 'ORDER'
    hdu.header['EXTVER'] = 3
    hdul.append(hdu)

    # Create HDU list.
    hdul = fits.HDUList(hdul)

    return hdul
