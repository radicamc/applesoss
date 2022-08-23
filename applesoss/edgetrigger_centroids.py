#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 15:35:32 2020

@author: albert

Functions necessary to locate the centroids of the NIRISS SOSS trace using the
edgetrigger algorithm.
"""

from astropy.io import fits
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import warnings

from applesoss.edgetrigger_utils import zero_roll, robust_polyfit, get_image_dim


def _plot_centroid(image, xtrace, ytrace):
    """Overplot the extracted trace positions on the image.

    Parameters
    ----------
    image : array[float]
        A 2D image of the detector.
    xtrace : array[float]
        The x coordinates of the trace to overplot on the image.
    ytrace : array[float]
        The y coordinates of the trace to overplot on the image.
    """

    nrows, ncols = image.shape

    if nrows == ncols:
        aspect = 1
        figsize = ncols/64, nrows/64
    else:
        aspect = 2
        figsize = ncols/64, nrows/32

    plt.figure(figsize=figsize)
    plt.title('Trace Centroids')

    plt.imshow(image, origin='lower', cmap='inferno', norm=colors.LogNorm(),
               aspect=aspect)
    plt.plot(xtrace, ytrace, lw=2, ls='--', c='black', label='Centroids')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Spatial Pixel', fontsize=14)
    plt.legend(fontsize=12)
    plt.xlim(-0.5, ncols - 0.5)
    plt.ylim(-0.5, nrows - 0.5)

    plt.tight_layout()
    plt.show()
    plt.close()


def _plot_centroids(image, centroids):
    """Visualize the trace extracted by get_soss_centroids().

    Parameters
    ----------
    image : array[float]
        A 2D image of the detector.
    centroids : dict
        A dictionary containg the trace, as returned by get_soss_centroids().
    """

    # Determine an appropriate figure size.
    nrows, ncols = image.shape

    if nrows == ncols:
        aspect = 1
        figsize = ncols/64, nrows/64
    else:
        aspect = 2
        figsize = ncols/64, nrows/32

    # Make a figure showing the trace for all 3 orders.
    plt.figure(figsize=figsize)
    plt.title('Trace Positions')

    plt.imshow(image, origin='lower', cmap='inferno', norm=colors.LogNorm(),
               aspect=aspect)

    tmp = centroids['order 1']
    plt.plot(tmp['X centroid'], tmp['Y centroid'], color='orange',
             label='Order 1')
    plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths']/2,
             color='orange')
    plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths']/2,
             color='orange')

    if 'order 2' in centroids:
        tmp = centroids['order 2']
        plt.plot(tmp['X centroid'], tmp['Y centroid'], color='black',
                 label='Order 2')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths']/2,
                 color='black')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths']/2,
                 color='black')

    if 'order 3' in centroids:
        tmp = centroids['order 3']
        plt.plot(tmp['X centroid'], tmp['Y centroid'], color='red',
                 label='Order 3')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] - tmp['trace widths']/2,
                 color='red')
        plt.plot(tmp['X centroid'], tmp['Y centroid'] + tmp['trace widths']/2,
                 color='red')

    plt.xlabel('Spectral Pixel', fontsize=14)
    plt.ylabel('Spatial Pixel', fontsize=14)
    plt.legend(fontsize=12)
    plt.xlim(-0.5, ncols - 0.5)
    plt.ylim(-0.5, nrows - 0.5)

    plt.tight_layout()
    plt.show()
    plt.close()


def edge_trigger(image, halfwidth=5, yos=1, verbose=False, outdir=None):
    """Detect the edges and center of the trace based on the minima and maxima
    of the derivative  of the columns, which is computed in a running window
    along the columns of the detector image

    Parameters
    ----------
    image : array[float]
        A 2D image of the detector.
    halfwidth : int
        Size of the window used when computing the derivatives.
    yos : int
        Oversampling factor of the image array along the y-direction.
    verbose : bool
        If set True some diagnostic plots will be made.
    outdir : str
        Directory to which to save results.

    Returns
    -------
    ytrace_max : array[float]
        Upper edge of the trace.
    ytrace_min : array[float]
        Lower edge of the trace.
    ytrace_best : array[float]
        Center of the trace.
    widths_best : array[float]
        Widths of the trace.
     """

    dimy, dimx = image.shape
    halfwidth = halfwidth * yos

    # Create coordinate arrays.
    xpix = np.arange(dimx)
    ypix = np.arange(dimy)
    _, ygrid = np.meshgrid(xpix, ypix)

    # Compute windowed slopes over the columns.
    slopevals = np.zeros_like(image)
    for irow in range(halfwidth, dimy-halfwidth):

        # Compute the window indices.
        ymin = irow - halfwidth
        ymax = irow + halfwidth + 1

        # Get the x and y data to find the slope to.
        datay = image[ymin:ymax, :]
        mask = np.isfinite(datay)
        # Need to set values NaN in y to NaN in x.
        datax = np.where(mask, ygrid[ymin:ymax, :], np.nan)

        # Compute the slope.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xmean = np.nanmean(datax, axis=0, keepdims=True)
            ymean = np.nanmean(datay, axis=0, keepdims=True)

        with np.errstate(invalid='ignore'):
            num = np.nansum((datax - xmean) * (datay - ymean), axis=0)
            denom = np.nansum((datax - xmean) ** 2, axis=0)
            slope = num / denom

        # Set slopes computed from < 3 datapoints to NaN.
        slopevals[irow, :] = np.where(np.sum(mask, axis=0) >= 3, slope, 0.)

    # Find the upper and lower bounds on the trace.
    args = np.nanargmax(slopevals, axis=0)
    vals = np.nanmax(slopevals, axis=0)
    ytrace_max = np.where(vals != 0, ypix[args], np.nan)

    args = np.nanargmin(slopevals, axis=0)
    vals = np.nanmin(slopevals, axis=0)
    ytrace_min = np.where(vals != 0, ypix[args], np.nan)

    # Scan through a range of trace widths.
    slopes_best = np.zeros_like(xpix)
    ytrace_best = np.zeros_like(xpix)
    widths_best = np.zeros_like(xpix)
    for width in range(18*yos, 27*yos):

        # Add the slope and its offset negative.
        comb = slopevals - zero_roll(slopevals, -width)

        # Find the maximum resulting slope.
        args = np.nanargmax(comb, axis=0)
        vals = np.nanmax(comb, axis=0)

        # Update the best values.
        mask = (vals > slopes_best)
        slopes_best = np.where(mask, vals, slopes_best)
        ytrace_best = np.where(mask, ypix[args], ytrace_best)
        widths_best = np.where(mask, width, widths_best)

    # Set the y position to NaN if the best slope was zero.
    ytrace_best = np.where(slopes_best != 0, ytrace_best + widths_best/2.,
                           np.nan)
    widths_best = np.where(slopes_best != 0, widths_best, np.nan)

    if verbose:
        nrows, ncols = image.shape

        plt.figure(figsize=(ncols/128, nrows/128))
        plt.title('Edge-trigger Trace Positions')

        plt.imshow(image, origin='lower', cmap='inferno',
                   norm=colors.LogNorm())
        plt.plot(ytrace_min, lw=2, ls='--', c='black', label='Edges')
        plt.plot(ytrace_max, lw=2, ls='--', c='black')
        plt.plot(ytrace_best, lw=2, c='black', label='Centroids')

        plt.xlabel('Spectral Pixel', fontsize=14)
        plt.ylabel('Spatial Pixel', fontsize=14)
        plt.legend(fontsize=12)

        plt.tight_layout()

        if outdir is not None:
            plt.savefig(outdir+'/edge_trigger_ymedian{:4.0f}.png'.format(np.nanmedian(ytrace_best)))
        plt.show()
        plt.close()

    return ytrace_max, ytrace_min, ytrace_best, widths_best


def get_centroids_edgetrigger(image, header=None, mask=None, poly_order=11,
                              halfwidth=5, mode='combined', verbose=False,
                              outdir=None):
    """Determine the x, y coordinates of the trace using the derivatives along
    the y-axis. Works for either order if there is no contamination.

    Parameters
    ----------
    image : array[float]
        A 2D image of the detector.
    header : astropy.io.fits.Header instance
        The header from one of the SOSS reference files.
    mask : array[bool]
        A boolean array of the same shape as image. Pixels corresponding to
        True values will be masked.
    poly_order : int, None
        Order of the polynomial to fit to the extracted trace positions.
    halfwidth : int
        Size of the window used when computing the derivatives.
    mode : str
        Which trace values to use. Can be 'bottomedge', 'topedge', 'mean' or
        'combined'.
    verbose : bool
        If set True some diagnostic plots will be made.
    outdir : str
        Directory to which to save results.

    Returns
    -------
    xtrace : array[float]
        X-coordinates of the trace.
    ytrace : array[float]
        Y-coordinates of the trace.
    tracewidth : array[float]
        Trace widths.
    param : array[float]
        Best fittinng polynomial parameters.
    """

    # If no mask was given use all pixels.
    if mask is None:
        mask = np.zeros_like(image, dtype='bool')

    # Call the script that determines the dimensions of the image.
    result = get_image_dim(image, header=header, verbose=verbose)
    dimx, dimy, xos, yos, xnative, ynative, padding, refpix_mask = result

    # Replace masked pixel values with NaNs.
    image_masked = np.where(mask | ~refpix_mask, np.nan, image)

    # Use edge trigger to compute the edges and center of the trace.
    fkwargs = dict(halfwidth=halfwidth, yos=yos, verbose=verbose,
                   outdir=outdir)
    edge_outs = edge_trigger(image_masked, **fkwargs)
    ytrace_max, ytrace_min, ytrace_best, widths_best = edge_outs

    # Use different y-positions depending on the mode parameter.
    if mode == 'bottomedge':
        ytrace = ytrace_max
    elif mode == 'topedge':
        ytrace = ytrace_min
    elif mode == 'mean':
        ytrace = (ytrace_min + ytrace_max)/2.
    elif mode == 'combined':
        ytrace = ytrace_best
    else:
        raise ValueError('Unknown mode: {}'.format(mode))

    # Fit the y-positions with a polynomial and use result as true y-positions.
    xtrace = np.arange(dimx)
    mask = np.isfinite(ytrace)

    # If no polynomial order was given return the raw measurements.
    if poly_order is None:
        param = []
    else:
        param = robust_polyfit(xtrace[mask], ytrace[mask], poly_order)
        ytrace = np.polyval(param, xtrace)

    # If verbose visualize the result.
    if verbose is True:
        _plot_centroid(image_masked, xtrace, ytrace)

    return xtrace, ytrace, widths_best, param


def build_mask_vertical(shape, xlims, mask_right=True, mask_between=True):
    """Mask along the vertical(s) given by xlims. If xlims contains 1 element
    masks pixels blue-wards or red-wards according to the value of mask_blue
    (and mask_between is ignored). If xlims contains 2 elements masks pixels
    between or outside these values according to the value of mask_between
    (and mask_blue is ignored).

    Parameters
    ----------
    shape : tuple[int]
        Tuple containing the intended shape of the mask array.
    xlims : list[float]
        The column indices to use as the limits of the masked area.
    mask_right : bool
        If True mask pixels to the right of xlims, otherwise mask to the left.
    mask_between : bool
        If True mask pixels between xlims, otherwise mask= outside.

    Returns
    -------
    mask : array[bool]
        A mask the removes a vertical region according to xlims.
    """

    dimy, dimx = shape

    # Create a coordinate grid.
    x = np.arange(dimx)
    y = np.arange(dimy)
    xgrid, _ = np.meshgrid(x, y)

    if np.size(xlims) == 1:
        # Mask blue-wards or red-wards of a single value.
        if mask_right:
            mask = xgrid >= xlims[0]
        else:
            mask = xgrid < xlims[0]
    elif np.size(xlims) == 2:
        # Mask between or exterior to two values.
        if mask_between:
            mask = (xgrid >= xlims[0]) & (xgrid < xlims[1])
        else:
            mask = (xgrid < xlims[0]) | (xgrid >= xlims[1])
    else:
        msg = 'xlims must be a list or array of up to two indices.'
        raise ValueError(msg)

    return mask


def build_mask_sloped(shape, point1, point2, mask_above=True, verbose=False):
    """Mask pixels above or below the boundary line defined by point1 and
    point2.

    Parameters
    ----------
    shape : tuple[int]
        Tuple containing the intended shape of the mask array.
    point1 : list[float]
        The first x, y pair defining the boundary line.
    point2 : list[float]
        The second x, y pair defining the boundary line.
    mask_above : bool
        If True mask pixels above the boundary line, else mask below.
    verbose : bool
        If True be verbose.

    Returns
    -------
    mask : array[bool]
        A mask the removes a diagonal region along the slope defined by point1
        and point2.
    """

    dimy, dimx = shape

    # Obtain the parameters of the line by fitting the point.
    xvals = np.array([point1[0], point2[0]])
    yvals = np.array([point1[1], point2[1]])
    param = np.polyfit(xvals, yvals, 1)

    # Compute the position of the line at every x position.
    xline = np.arange(dimx)
    yline = np.polyval(param, xline)

    if verbose:
        print('line fit param:', param)

    # Create a coordinate grid.
    x = np.arange(dimx)
    y = np.arange(dimy)
    _, ygrid = np.meshgrid(x, y)

    # Mask pixels above or below the boundary line.
    if mask_above:
        mask = (ygrid - yline) >= 0
    else:
        mask = (ygrid - yline) < 0

    return mask


def build_mask_256(subarray='SUBSTRIP256', apex_order1=None):
    """Restrict the analysis to a (N, 2048) section of the image, where N is
    256 or less. Normally this only applies to the FULL subarray, masking
    everything but the SUBSTRIP256 region. When apex_order1 is given rows from
    apex_order1 - 40 to apex_order1 + 216 are kept instead.

    Parameters
    ----------
    subarray : str
        The subarray for which to build a mask.
    apex_order1 : float
        The y-position of the order1 apex at 1.3 microns, in the given
        subarray.

    Returns
    -------
    mask_256 : array[bool]
        A mask that removes any area not related to the trace of the target.
    """

    dimx = 2048

    # Check the subarray value and set dimy accordingly.
    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    if apex_order1 is None:
        apex_order1 = 40  # Assuming SUBSTRIP256.

        if subarray == 'FULL':
            apex_order1 += 1792

        if subarray == 'SUBSTRIP96':
            apex_order1 += -10

    # Round the apex value to the nearest integer.
    apex_order1 = int(apex_order1)

    # Prepare the mask array.
    mask_256 = np.ones((dimy, dimx), dtype='bool')

    # Keep only the 256 region around the apex_order1 value.
    # In SUBSTRIP256 the apex would be at y ~ 40.
    rowmin = np.maximum(apex_order1 - 40, 0)
    rowmax = np.minimum(apex_order1 + 216, dimy)
    mask_256[rowmin:rowmax, :] = False

    return mask_256


def build_mask_trace(ytrace, subarray='SUBSTRIP256', halfwidth=30,
                     extend_below=False, extend_above=False):
    """Mask out the trace in a given subarray based on the y-positions
    provided. A band of pixels around the trace position of
    width = 2*halfwidth will be masked. Optionally extend_above and
    extend_below can be used to mask all pixels above or below the trace.

    Parameters
    ----------
    ytrace : array[float]
        The trace y-position at each column, must have shape = (2048,).
    subarray : str
        The subarray for which to build a mask.
    halfwidth : float
        The size of the window to mask around the trace.
    extend_below : bool
        If True mask all pixels above the trace.
    extend_above : bool
        If True mask all pixels below the trace.

    Returns
    -------
    mask_trace : array[bool]
        A mask that removes an area centered on the given trace positions.
    """

    dimx = 2048

    # Check the shape of the y-positions.
    if np.shape(ytrace) != (dimx,):
        msg = 'ytrace must have shape (2048,)'
        raise ValueError(msg)

    # Check the subarray value and set dimy accordingly.
    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    # Cannot both be True, that would mask everything.
    if extend_below and extend_above:
        msg = 'Only one of extend_below, extend_above should be used.'
        raise ValueError(msg)

    # Create a coordinate grid.
    x = np.arange(dimx)
    y = np.arange(dimy)
    _, ygrid = np.meshgrid(x, y)

    # Mask the pixels within a halfwidth of the trace center.
    mask_trace = np.abs(ygrid - ytrace) < halfwidth

    # If True mask all pixels below the trace center.
    if extend_below:
        mask_below = (ygrid - ytrace) < 0
        mask_trace = mask_trace | mask_below

    # If True mask all pixels above the trace center.
    if extend_above:
        mask_above = (ygrid - ytrace) >= 0
        mask_trace = mask_trace | mask_above

    return mask_trace


def build_mask_order2_contaminated(ytrace_o1, ytrace_o3,
                                   subarray='SUBSTRIP256', halfwidth_o1=25,
                                   halfwidth_o3=15, xlim=150):
    """Build a mask that isolates the contaminated part of the order 2 trace.
    This is done by masking the order 1 trace and averything below, the order
    2 trace and everything above and all pixels blue-ward (to the right) of
    xlim.

    Parameters
    ----------
    ytrace_o1 : array[float]
        Y position of the order 1 trace at every column.
    ytrace_o3 : array[float]
        Y position of the order 3 trace at every column.
    subarray : str
        The subarray for which to build a mask.
    halfwidth_o1 : float
        The size of the window to mask around the order 1 trace.
    halfwidth_o3 : float
        The size of the window to mask around the order 3 trace.
    xlim : float
        The boundary for masking pixels blue-wards (to the right).

    Returns
    -------
    mask : array[bool]
        A mask that removes everything but the contaminated part of the order
        2 trace.
    """

    dimx = 2048

    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    # Mask the order 1 trace and everything below.
    mask_trace_o1 = build_mask_trace(ytrace_o1, subarray=subarray,
                                     halfwidth=halfwidth_o1,
                                     extend_below=True)

    # Mask the order 3 trace and everything above.
    mask_trace_o3 = build_mask_trace(ytrace_o3, subarray=subarray,
                                     halfwidth=halfwidth_o3,
                                     extend_above=True)

    # Mask all pixels blue-ward of xlim.
    mask_blue = build_mask_vertical((dimy, dimx), xlims=[xlim],
                                    mask_right=True)

    # Combine the masks.
    mask = mask_trace_o1 | mask_trace_o3 | mask_blue

    return mask


def build_mask_order2_uncontaminated(ytrace_o1, ytrace_o3,
                                     subarray='SUBSTRIP256', halfwidth_o1=25,
                                     halfwidth_o3=15, xlims=None, point1=None,
                                     point2=None, apex_order1=None):
    """Build a mask that isolates the uncontaminated part of the order 2 trace.
    This is done by masking the order 1 trace and averything below, the order
    2 trace and everything above, all pixels outside of the range defined by
    xlims and all pixels below the line defined by point 1 and point 2.

    Parameters
    ----------
    ytrace_o1 : array[float]
        Y position of the order 1 trace at every column.
    ytrace_o3 : array[float]
        Y position of the order 3 trace at every column.
    subarray : str
        The subarray for which to build a mask.
    halfwidth_o1 : float
        The size of the window to mask around the order 1 trace.
    halfwidth_o3 : float
        The size of the window to mask around the order 3 trace.
    xlims : list[float]
        X-pixel limits.
    point1 : list[float]
        The first x, y pair defining the boundary line.
    point2 : list[float]
        The second x, y pair defining the boundary line.
    apex_order1 : float, np.ndarray[float]
        The y-position of the order1 apex at 1.3 microns, in the given
        subarray.

    Returns
    -------
    mask : array[bool]
        A mask that removes everything but the uncontaminated part of the
        order 2 trace.
    """

    dimx = 2048

    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    if xlims is None:
        xlims = [700, 1800]

    if (point1 is None) ^ (point2 is None):
        msg = 'point1 and point2 must both be None or both be set.'
        raise ValueError(msg)

    elif (point1 is None) & (point2 is None):
        # If no points were given use default values.
        point1 = [1249, 31]  # Assuming SUBSTRIP256.
        point2 = [1911, 253]  # Assuming SUBSTRIP256.

        if subarray == 'FULL':
            point1[1] += 1792
            point2[1] += 1792

        if subarray == 'SUBSTRIP96':
            point1[1] += -10
            point2[1] += -10

        # If apex_order1 was given shift the points as needed.
        if apex_order1 is not None:
            apex_default = 40  # Assuming SUBSTRIP256.

            if subarray == 'FULL':
                apex_default += 1792
            if subarray == 'SUBSTRIP96':
                apex_default += -10

            # Shift points based on apex_order1.
            offset = apex_order1 - apex_default
            point1[1] += offset
            point2[1] += offset
    else:
        msg = ('Using user-provided values for point1 and point2, '
               'apex_order1 will be ignored.')
        print(msg)

    # Mask the order 1 trace and everything below.
    mask_trace_o1 = build_mask_trace(ytrace_o1, subarray=subarray,
                                     halfwidth=halfwidth_o1,
                                     extend_below=True)

    # Mask the order 3 trace and everything above.
    mask_trace_o3 = build_mask_trace(ytrace_o3, subarray=subarray,
                                     halfwidth=halfwidth_o3,
                                     extend_above=True)

    # Mask what is on the left side where orders 1 and 2 are well blended
    mask_vertical = build_mask_vertical((dimy, dimx), xlims,
                                        mask_between=False)

    # Mask the corner below the order 2 trace to remove the wings of the
    # order 1 trace.
    mask_sloped = build_mask_sloped((dimy, dimx), point1, point2,
                                    mask_above=False)

    # Combine the masks.
    mask = (mask_trace_o1 | mask_trace_o3 | mask_vertical | mask_sloped)

    return mask


def build_mask_order3(subarray='SUBSTRIP256', xlim=None, point1=None,
                      point2=None, apex_order1=None):
    """Builds a mask that isolates the order 3 trace.
    This done by masking all pixels blue-wards (to the right) of xlim where the
    order 3 transmission goes to zero, and all pixels below the line defined
    by point1 and point2 (the order1 trace and order 2 trace).


    Parameters
    ----------
    subarray : str
        The subarray for which to build a mask.
    xlim : float
        The boundary for masking pixels blue-ward (to the right).
    point1 : list[float]
        The first x, y pair defining the boundary line.
    point2 : list[float]
        The second x, y pair defining the boundary line.
    apex_order1 : float, np.ndarray[float]
        The y-position of the order1 apex at 1.3 microns, in the given
        subarray.

    Returns
    -------
    mask : array[bool]
        A mask that removes everything but the order 3 trace.
    """

    dimx = 2048

    if subarray == 'FULL':
        dimy = 2048
    elif subarray == 'SUBSTRIP96':
        dimy = 96
    elif subarray == 'SUBSTRIP256':
        dimy = 256
    else:
        msg = 'Unknown subarray: {}'
        raise ValueError(msg.format(subarray))

    if subarray == 'SUBSTRIP96':
        # Create an empty mask.
        mask = np.zeros((dimy, dimx), dtype='bool')

        # Nothing to be done because order 3 can not be present.
        print('Warning. No mask produced for order 3 when subarray=SUBSTRIP96')

        return mask

    if xlim is None:
        xlim = 700

    if (point1 is None) ^ (point2 is None):
        msg = 'point1 and point2 must both be None or both be set.'
        raise ValueError(msg)

    elif (point1 is None) & (point2 is None):
        # If no points were given use default values.
        point1 = [0, 132]  # Assuming SUBSTRIP256.
        point2 = [1000, 163]  # Assuming SUBSTRIP256.

        if subarray == 'FULL':
            point1[1] += 1792
            point2[1] += 1792

        if subarray == 'SUBSTRIP96':
            point1[1] += -10
            point2[1] += -10

        # If apex_order1 was given shift the points as needed.
        if apex_order1 is not None:

            apex_default = 40  # Assuming SUBSTRIP256.

            if subarray == 'FULL':
                apex_default += 1792

            if subarray == 'SUBSTRIP96':
                apex_default += -10

            # Shift points based on apex_order1.
            offset = apex_order1 - apex_default
            point1[1] += offset
            point2[1] += offset

    else:
        msg = ('Using user-provided values for point1 and point2, '
               'apex_order1 will be ignored.')
        print(msg)

    # Check how close the boundary line is to the top of the subarray.
    if point1[1] > (dimy - 25 - 10):
        msg = ('Warning: masking for order 3 leaves too little of '
               'order 3 to fit position.')
        print(msg)

    # Mask everything beyond where the order 3 transmission approaches zero.
    mask_vertical = build_mask_vertical((dimy, dimx), [xlim], mask_right=True)

    # Mask everything below order 3.
    mask_sloped = build_mask_sloped((dimy, dimx), point1, point2,
                                    mask_above=False)

    # Combine the masks.
    mask = mask_vertical | mask_sloped

    return mask


def wavelength_calibration(tracetable, xpos, order=1):
    """Find the wavelengths corresponding to a set of x-positions using the
    trace table reference file.

    Parameters
    ----------
    tracetable : str
        Path to SOSS tracetable reference file.
    xpos : array[float]
        The array of x-positions to calibrate.
    order : int
        The trace order the x-positions correspond to.

    Returns
    -------
    wavelengths : array[float]
        An array of wavelengths corresponding to xpos.
    """

    try:
        # USE THE REFERENCE FILE IF IT CAN BE FOUND
        # Read the wavelength vs x-position relation from the reference file.
        tt = fits.getdata(tracetable, order)
        ref_wavelengths, ref_xpos = tt['WAVELENGTH'], tt['X']

        # Sort so the reference positions are in ascending order.
        args = np.argsort(ref_xpos)
        ref_xpos, ref_wavelengths = ref_xpos[args], ref_wavelengths[args]

        # Find the wavelengths corresponding to the input array via
        # interpolation.
        wavelengths = np.interp(xpos, ref_xpos, ref_wavelengths)
    except:
        # USE AN APPROXIMATE WAVELENGTH CALIBRATION OTHERWISE
        if order == 1:
            dispersion = -0.9718  # nm/pixel
            w0 = 2.833
        elif order == 2:
            dispersion = -0.467
            w0 = 1.423
        else:
            dispersion = -0.310
            w0 = 0.956
        wavelengths = w0 + xpos * (dispersion / 1000)

    return wavelengths


def calibrate_widths(tracetable, width_o1, width_o2=None, width_o3=None,
                     verbose=False, outdir=None):
    """Fit an exponential function to the wavelength-width relation, for use
    obtaining the contaminated order 2 trace positions.

    Parameters
    ----------
    tracetable : str
        Path to SOSS tracetable reference file.
    width_o1 : array[float]
        The order 1 trace width at each column, must have shape = (2048,).
    width_o2 : array[float]
        The order 2 trace width at each column, must have shape = (2048,).
    width_o3 : array[float]
        The order 3 trace width at each column, must have shape = (2048,).
    verbose : bool
        If True some diagnostic plots will be made.
    outdir : str
        Directry to which to save results.

    Returns
    -------
    pars_width : list[float]
        List containing the best-fit parameters for the wavelength-width
        relation.
    """

    dimx = 2048

    # Check the shapes of the widths.
    if np.shape(width_o1) != (dimx,):
        msg = 'width_o1 must have shape (2048,)'
        raise ValueError(msg)

    if width_o2 is not None:
        if np.shape(width_o2) != (dimx,):
            msg = 'width_o2_uncont must have shape (2048,)'
            raise ValueError(msg)
    else:
        width_o2 = np.full(dimx, fill_value=np.nan)

    if width_o3 is not None:
        if np.shape(width_o3) != (dimx,):
            msg = 'width_o3_uncont must have shape (2048,)'
            raise ValueError(msg)
    else:
        width_o3 = np.full(dimx, fill_value=np.nan)

    # Convert pixel positions to wavelengths for each order.
    x = np.arange(dimx)
    lba_o1 = wavelength_calibration(tracetable, x, order=1)
    lba_o2 = wavelength_calibration(tracetable, x, order=2)
    lba_o3 = wavelength_calibration(tracetable, x, order=3)

    # Join data from different orders.
    lba_all = np.concatenate((lba_o1, lba_o2, lba_o3), axis=None)
    width_all = np.concatenate((width_o1, width_o2, width_o3), axis=None)

    # Fit the wavelength vs width of order 1 and 2 using an exponential model.
    mask = np.isfinite(width_all) & np.isfinite(lba_all)
    pars_width = robust_polyfit(np.log(lba_all[mask]),
                                np.log(width_all[mask]), 1)

    # Make a figure of the trace width versus the wavelength
    if verbose:
        # Evalaute the best-fit model.
        lba_fit = np.linspace(np.nanmin(lba_all), np.nanmax(lba_all), 101)
        w0, m = np.exp(pars_width[1]), pars_width[0]  # w = w0 * lba^m
        width_fit = w0 * lba_fit ** m

        # Make the figure.
        plt.figure(figsize=(8, 5))

        plt.scatter(lba_o1, width_o1, marker=',', s=1, color='red',
                    label='Order 1')

        if np.any(np.isfinite(width_o2)):
            plt.scatter(lba_o2, width_o2 + 0.05, marker=',', s=1,
                        color='orange', label='Order 2')

        if np.any(np.isfinite(width_o3)):
            plt.scatter(lba_o3, width_o3 + 0.10, marker=',', s=1, color='navy',
                        label='Order 3')

        plt.plot(lba_fit, width_fit, color='black', linewidth=5,
                 label='Joint Fit:\nwidth = {:6.2F} $\\lambda**({:6.4F})$'.format(w0, m))

        plt.xlabel('Wavelength (microns)', fontsize=12)
        plt.ylabel('Trace Width (pixels)', fontsize=12)
        plt.legend(fontsize=12)

        plt.tight_layout()

        if outdir is not None:
            plt.savefig(outdir+'/soss_centroids_calibrate_width.png')
        plt.show()
        plt.close()

    return pars_width


def get_soss_centroids(image, tracetable, mask=None, subarray='SUBSTRIP256',
                       halfwidth=2, poly_orders=None, apex_order1=None,
                       calibrate=True, verbose=False, outdir=None):
    """Determine the traces positions on a real image (native size) with as
    few assumptions as possible using the 'edge trigger' method.
    The algorithm assumes:
    1) The brightest order is order 1 and the target order 1 is the brightest
        of all order 1 traces present.
    2) Order 2 has a minimum in transmission between ~1.0 and ~1.2 microns.
    3) Order 2 widths are the same as order 1 width for the same wavelengths.

    Parameters
    ----------
    image : array[float]
        A 2D image of the detector.
    tracetable : str
        Path to SOSS tracetable reference file.
    mask : array[bool]
        A boolean array of the same shape as image. Pixels corresponding to
        True values will be masked.
    subarray : str
        The subarray for which to build a mask.
    halfwidth : int
        The size of the window used when computing the derivatives of the
        'edge trigger' method.
    poly_orders : dict
        Dictionary of polynomial orders to fit to the extracted trace
        positions for each spectral order.
    apex_order1 : float
        The y-position of the order1 apex at 1.3 microns, in the given
        subarray. A rough estimate is sufficient as it is only used to mask
        rows when subarray='FULL' to ensure that the target of interest is
        detected instead of a field target.
    calibrate : bool
        If True model the wavelength trace width relation, otherwise use the
        CV3 parameters. Default is True.
    verbose : bool
        If set True some diagnostic plots will be made. Default is False.
    outdir : str
        Directory to which to save results.

    Returns
    -------
    trace_dict : dict
        A dictionary containing the trace x, y, width and polynomial fit
        parameters for each order.
    """

    default_orders = {'order 1': 11,
                      'order 2': 5,
                      'order 3': 3}

    if poly_orders is not None:
        default_orders = {**default_orders, **poly_orders}

    # Initialize output dictionary.
    centroids = dict()

    # Build a mask that restricts the analysis to a SUBSTRIP256-like region
    # centered on the target trace.
    mask_256 = build_mask_256(subarray=subarray, apex_order1=apex_order1)

    # Combine the subsection mask with the user specified mask.
    if mask is not None:
        mask_256 = mask_256 | mask

    if verbose & (outdir is not None):
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_256, np.nan, image)
        hdu.writeto(outdir+'/mask_256.fits', overwrite=True)

    # Get the order 1 trace position.
    result = get_centroids_edgetrigger(image, mask=mask_256,
                                       poly_order=default_orders['order 1'],
                                       halfwidth=halfwidth, mode='combined',
                                       verbose=verbose, outdir=outdir)

    x_o1, y_o1, w_o1, par_o1 = result

    # Add parameters to output dictionary.
    o1_dict = dict()
    o1_dict['X centroid'] = x_o1
    o1_dict['Y centroid'] = y_o1
    o1_dict['trace widths'] = w_o1
    o1_dict['poly coefs'] = par_o1
    centroids['order 1'] = o1_dict

    # For SUBSTRIP96 only the order 1 can be measured.
    if subarray == 'SUBSTRIP96':

        if verbose:

            # Make a figure showing the order 1 trace.
            _plot_centroids(image, centroids)

        return centroids

    # Update the order1 apex based on the extracted trace.
    apex_order1 = np.nanmin(y_o1)

    # Make a mask to isolate the order 3 trace and combine it with the
    # user-specified mask.
    mask_o3 = build_mask_order3(subarray=subarray, apex_order1=apex_order1)

    if mask is not None:
        mask_o3 = mask_o3 | mask

    if verbose & (outdir is not None):
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o3, np.nan, image)
        hdu.writeto(outdir+'/mask_o3.fits', overwrite=True)

    # Get the order 3 trace position.
    result = get_centroids_edgetrigger(image, mask=mask_o3,
                                       poly_order=default_orders['order 3'],
                                       halfwidth=halfwidth, mode='combined',
                                       verbose=verbose, outdir=outdir)

    x_o3, y_o3, w_o3, par_o3 = result

    # Add parameters to output dictionary.
    o3_dict = dict()
    o3_dict['X centroid'] = x_o3
    o3_dict['Y centroid'] = y_o3
    o3_dict['trace widths'] = w_o3
    o3_dict['poly coefs'] = par_o3
    centroids['order 3'] = o3_dict

    # Make masks for the second order trace - split in two segments:
    # A) Uncontaminated region 700 < x < 1800 - fit both edges combined
    # (default).
    # B) Contaminated region (x = 0-200) - fit only the top edge.

    # Make a mask to isolate the uncontaminated order 2 trace and combine it
    # with the user-specified mask.
    mask_o2_uncont = build_mask_order2_uncontaminated(y_o1, y_o3,
                                                      subarray=subarray,
                                                      apex_order1=apex_order1)

    if mask is not None:
        mask_o2_uncont = mask_o2_uncont | mask

    if verbose & (outdir is not None):
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_uncont, np.nan, image)
        hdu.writeto(outdir+'/mask_o2_uncont.fits', overwrite=True)

    # Get the raw trace positions for the uncontaminated part of the order 2
    # trace.
    result = get_centroids_edgetrigger(image, mask=mask_o2_uncont,
                                       poly_order=None, halfwidth=halfwidth,
                                       mode='combined', verbose=verbose,
                                       outdir=outdir)

    x_o2_uncont, y_o2_uncont, w_o2_uncont, par_o2_uncont = result

    if calibrate:
        pars_width = calibrate_widths(tracetable, w_o1, w_o2_uncont,
                                      verbose=verbose, outdir=outdir)

    else:
        # Use pre-computed parameters from the CV3 deepstack.
        pars_width = [-0.20711659, 3.16387517]

    w0, m = np.exp(pars_width[1]), pars_width[0]  # w = w0 * lba^m

    # Make a mask to isolate the contaminated order 2 trace and combine it
    # with the user-specified mask.
    mask_o2_cont = build_mask_order2_contaminated(y_o1, y_o3,
                                                  subarray=subarray)

    if mask is not None:
        mask_o2_cont = mask_o2_cont | mask

    if verbose & (outdir is not None):
        hdu = fits.PrimaryHDU()
        hdu.data = np.where(mask_o2_cont, np.nan, image)
        hdu.writeto(outdir+'/mask_o2_cont.fits', overwrite=True)

    # Get the raw top-edge poistions of the contaminated order 2 trace.
    result = get_centroids_edgetrigger(image, mask=mask_o2_cont,
                                       poly_order=None, halfwidth=halfwidth,
                                       mode='topedge', verbose=verbose,
                                       outdir=outdir)

    x_o2_top, y_o2_top, w_o2_top, par_o2_top = result

    # Convert pixel positions to wavelengths for order 2.
    lba_o2_top = wavelength_calibration(tracetable, x_o2_top, order=2)

    # Use the wavelength width relation to obtain the order 2 trace width.
    w_o2_cont = np.where(np.isfinite(w_o2_top), w0 * lba_o2_top**m, np.nan)

    # Finally combine the top-edge positions and the width to get an estimate
    # of the trace center.
    x_o2_cont = np.copy(x_o2_top)
    y_o2_cont = y_o2_top - w_o2_cont/2.

    # Combine the trace positions from the uncontaminated and contaminated
    # sections.
    mask_comb = np.isfinite(y_o2_uncont)
    x_o2 = np.where(mask_comb, x_o2_uncont, x_o2_cont)
    y_o2 = np.where(mask_comb, y_o2_uncont, y_o2_cont)
    w_o2 = np.where(mask_comb, w_o2_uncont, w_o2_cont)

    # Fit the combined order 2 trace position with a polynomial.
    mask_fit = np.isfinite(x_o2) & np.isfinite(y_o2)
    if default_orders['order 2'] is None:
        par_o2 = []
    else:
        par_o2 = robust_polyfit(x_o2[mask_fit], y_o2[mask_fit],
                                default_orders['order 2'])
        y_o2 = np.polyval(par_o2, x_o2)

    if verbose:
        # Determine an appropriate figure size.
        nrows, ncols = image.shape

        if subarray == 'FULL':
            aspect = 1
            figsize = ncols/64, nrows/64
        else:
            aspect = 2
            figsize = ncols/64, nrows/32

        # Make a figure showing how the order 2 trace was built from segments
        # A and B.
        plt.figure(figsize=figsize)
        plt.title('Order 2 Trace Positions')

        plt.imshow(image, origin='lower', cmap='inferno',
                   norm=colors.LogNorm(), aspect=aspect)

        plt.plot(x_o2_cont, y_o2_cont, color='red', label='Contaminated')
        plt.plot(x_o2_uncont, y_o2_uncont, color='navy',
                 label='Uncontaminated')
        plt.plot(x_o2, y_o2, color='black', label='Polynomial Fit')

        plt.xlabel('Spectral Pixel', fontsize=14)
        plt.ylabel('Spatial Pixel', fontsize=14)
        plt.legend(fontsize=12)
        plt.xlim(-0.5, ncols - 0.5)
        plt.ylim(-0.5, nrows - 0.5)

        plt.tight_layout()

        if outdir is not None:
            plt.savefig(outdir+'/soss_centroid_order2tracepositions.png')
        plt.show()
        plt.close()

    # Add parameters to output dictionary.
    o2_dict = dict()
    o2_dict['X centroid'] = x_o2
    o2_dict['Y centroid'] = y_o2
    o2_dict['trace widths'] = w_o2
    o2_dict['poly coefs'] = par_o2
    centroids['order 2'] = o2_dict

    if verbose:

        # Make a figure showing the trace for all orders.
        _plot_centroids(image, centroids)

    return centroids
