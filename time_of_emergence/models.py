""" This module implements the algorithm of `Hawkins and Sutton (2012)`_ to
estimate time of emergence for a given signal given background (internal)
climate variability.

.. _Hawkins and Sutton (2012): http://dx.doi.org/10.1029/2011GL050087

"""
from functools import partial
from darpy import global_avg, shift_lons

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt


def periodic_x_connectivity_matrix(n_x, n_y, order='C'):
    """ Compute the adjaceny/connectivity matrix for an n_x by n_y array
    assuming that the data is periodic in the x-direction. """

    n_tot = n_x*n_y
    conn_matrix = np.zeros((n_tot, n_tot), dtype=np.uint0)

    # def loc_to_global(i, j, n_x=n_x, n_y=n_y):
    #     """ Given the coordinate of an element in an n_x x n_y matrix,
    # yield its unraveled (column-oriented) index """
    #     return j*n_x + i
    def loc_to_global(i, j, n_x=n_x, n_y=n_y):
        return np.ravel_multi_index([[i,], [j,]], (n_x, n_y), order=order)

    # def global_to_loc(i, n_x=n_x, n_y=n_y):
    #     """ Given the unraveled (column-oriented) index of an n_x*n_y
    # length vector, yield the corresponding coordinate for the raveled
    # n_x x n_y array
    # """
    #     return i % n_x, i // n_x
    def global_to_loc(i, n_x=n_x, n_y=n_y):
        return np.unravel_index([i, ], (n_x, n_y), order=order)

    for idx in range(n_tot):

        i, j = global_to_loc(idx)

        if i == 0:
            im1 = i + n_x - 1
            # im1 = None
            ip1 = i + 1
        elif i == (n_x - 1):
            im1 = i - 1
            ip1 = i - n_x + 1
            # ip1 = None
        else:
            im1 = i - 1
            ip1 = i + 1

        if j == 0:
            jm1 = None
            jp1 = j + 1
        elif j == (n_y - 1):
            jm1 = j - 1
            jp1 = None
        else:
            jm1 = j - 1
            jp1 = j + 1

        for ni, nj in [
            (i, jm1), # ABOVE
            (im1, j), # LEFT
            (i,   j), # THIS
            (ip1, j), # RIGHT
            (i, jp1)  # BELOW
        ]:
            if (ni is None) or (nj is None):
                continue
            nidx = loc_to_global(ni, nj)
            conn_matrix[idx, nidx] += 1

    return conn_matrix

def calc_signal_vs_noise(model, years=(1980, 2150), baseline=(1980, 2010)):
    """ Given a fitted Hawkins and Sutton (2012) model, estimate the
    signal vs noise statistics from the smoothed, predicted timeseries.

    """

    years = range(*years)
    baseline_slice = slice(*baseline)

    Ss = [model.predict(year) for year in years]
    S = xr.concat(Ss, 'year')
    S['year'] = (['year'], years)

    noise = model.data.sel(**{model.time_dim: baseline_slice}).std(model.time_dim)
    SN = S  / noise
    SN['year'] = (['year'], years)
    signal_ds = xr.Dataset({'S': S, 'S_N': SN, 'N': noise})

    return signal_ds


def calc_time_of_emergence(signal_ds, ratios=[1, 2, 3], mask=False):
    """ Given the signal vs noise statistics from `calc_signal_vs_noise`,
    estimate time of emergence for different S/N ratios. If `mask` is enabled,
    then clip all values which are beyond the maximum year predicted in
    **signal_ds**.

    """
    template = signal_ds['S'].isel(year=0).data
    toe_s = np.zeros_like(template)

    toe_das = []
    for ratio in ratios:
        toe_s = np.zeros_like(template)
        for year in signal_ds.year:
            _sn_year = signal_ds['S_N'].sel(year=year).data
            toe_s[(toe_s == 0) & (_sn_year > ratio)] = year
        if mask:
            toe_s = np.ma.masked_equal(toe_s, 0)
        else:
            toe_s[toe_s == 0] = year
        toe_da = xr.DataArray(toe_s,
                              dims=['lat', 'lon'],
                              coords=[signal_ds.lat, signal_ds.lon])
        toe_das.append(toe_da)

    toe_ds = xr.concat(toe_das, dim=pd.Index(ratios, name='ratio'))
    toe_ds.name = 'TOE'
    toe_ds.attrs['max_year'] = year

    return toe_ds


class HawkinsSutton2012(object):

    def __init__(self, data, degree=4, time_dim='time'):

        self._data = data
        self._global_avg_data = global_avg(self._data)

        self.degree = degree
        self.time_dim = time_dim

        self._fitted_model = None

    @property
    def data(self):
        return self._data

    def fit(self):
        if self._fitted_model is not None:
            raise ValueError("Model has already been fit")

        # Fit smoothed regression of global mean
        y = self._global_avg_data.data
        x = self._global_avg_data[self.time_dim].data
        p = np.polyfit(x, y, self.degree)
        xs = range(x[0], x[-1])
        y_hat = [self._model(xi, p) for xi in xs]
        fit_model = partial(self._model, p=p)

        # Compute grid cell regressions to smoothed global mena
        stacked = self._data.stack(cell=['lat', 'lon'])
        x = stacked[self.time_dim].data
        y = stacked.data
        # TODO: Generalize to sklearn or statsmodels so that we can have
        #       statistics on the regression fit.
        p = np.polyfit(x, y, 1)
        alphas, betas = p

        stacked['α'] = (['cell', ], alphas)
        stacked['β'] = (['cell', ], betas)
        self._fitted_model = stacked.unstack('cell')
        self.α = self._fitted_model['α']
        self.β = self._fitted_model['β']

    def predict(self, x):
        return self.α*x + self.β

    @staticmethod
    def _model(x, p):
        """ Polynomial model; p should be a vector of coefficients
        corresponding in *decreasing* order (e.g. p[-1] is the intercept)

        """
        y_hat = 0
        for i, pi in enumerate(reversed(p)):
            y_hat += x**i * pi
        return y_hat

    def plot_coeffs(self, alpha_range=None, beta_range=None, **kws):
        """ Plot maps of alpha and beta fields on same figure """

        import cartopy.crs as ccrs

        size, aspect = 3., 2.2
        nrows, ncols = 1, 2
        width, height = size*aspect*ncols, size*nrows

        fig, axs = plt.subplots(nrows, ncols, figsize=(width, height),
                                subplot_kw={'projection': ccrs.Robinson()})

        default_kws = dict(transform=ccrs.PlateCarree(), robust=True)
        alpha_cmap_kws = default_kws.copy()
        if alpha_range is not None:
            alpha_cmap_kws.update({'vmin': alpha_range[0],
                                   'vmax': alpha_range[1]})
        self.α.plot.imshow(ax=axs[0], **alpha_cmap_kws)

        beta_cmap_kws = default_kws.copy()
        if beta_range is not None:
            beta_cmap_kws.update({'vmin': beta_range[0],
                                  'vmax': beta_range[1]})
        self.β.plot.imshow(ax=axs[1], **beta_cmap_kws)

        for ax in axs:
            ax.coastlines()
            ax.set_aspect('auto', 'box-forced')
            ax.set_title("")

        plt.tight_layout()

        return fig, axs

    def plot_prediction_check(self, x, dx=0):

        from matplotlib.gridspec import GridSpec
        import cmocean
        import cartopy.crs as ccrs

        # Set up our plotting grid / axes
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 4)
        subplot_kw = dict(projection=ccrs.PlateCarree(), aspect='auto')

        axl = fig.add_subplot(gs[0, 0:2], **subplot_kw)
        axc = fig.add_subplot(gs[1, 1:3], **subplot_kw)
        axr = fig.add_subplot(gs[0, 2:], **subplot_kw)
        axs = [axl, axc, axr]

        # Compute mean centered around a given year
        if not dx:
            predicted = self.predict(x)
            actual = self._fitted_model.sel(**{self.time_dim: x})
            t_label = "{}".format(x)
        else:
            xlo, xhi = x-dx, x+dx
            all_predicted = xr.concat([self.predict(xi) for xi in range(xlo, xhi)], 'time')
            predicted = all_predicted.mean(self.time_dim)
            actual = (
                self._fitted_model.sel(time=slice(xlo, xhi))
                .mean(self.time_dim)
            )
            t_label = "{} $\pm$ {}".format(x, dx)
        diff = 100*(predicted - actual)/actual

        cmap_kws = dict(vmin=0, vmax=4, cmap=cmocean.cm.deep)
        predicted.plot.imshow(ax=axl, transform=ccrs.PlateCarree(),
                              **cmap_kws)
        axl.set_title("S(t={})".format(t_label))
        actual.plot.imshow(ax=axr, transform=ccrs.PlateCarree(),
                           **cmap_kws)
        axr.set_title("Modeled Anomaly (t={})".format(t_label))
        cmap_kws = dict(vmin=-250, vmax=250, cmap=cmocean.cm.balance)
        diff.plot.imshow(ax=axc, transform=ccrs.PlateCarree(),
                         **cmap_kws)
        axc.set_title("Relative difference (%)\nS vs Modeled")

        for ax in axs:
            ax.coastlines()
            ax.set_aspect('auto', 'box-forced')

        plt.tight_layout()

        return fig, axs
