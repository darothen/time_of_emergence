import numpy as np
import xarray as xr

def flatten_times(ds, time='time', aux_times='ic'):
    """ Given a Dataset with multiple realizations along the same timeseries
    dimension, unravel these realizations and linearly increment the time to
    produce a longer, flat timeseries. """

    nt = len(ds[time]) // 12

    _dss = []
    for i in range(len(ds[aux_times])):
        ds_aux_i = ds.isel(**{aux_times: i})

        # NOTE: Assume that our time offset will be in years, and that we can
        #       safely cast the existing timeseries to monthly values
        delta = np.timedelta64(nt*i, 'Y')
        ds_aux_i[time].values = \
            ds_aux_i[time].values.astype('datetime64[M]') + delta

        _dss.append(ds_aux_i)

    # Concatenate along the original time dimension
    ds_flat = xr.concat(_dss, time)

    return ds_flat


def fgm_unstack_years(data, monthly=False):
    """ "Un-stack" the three decadal slices in a consolidated FGM simulation
    output. If 'monthly' is passed, assumes that the data is monthly; else,
    annual averages. """

    dec_dict = {}
    decs = data.dec.values.astype(str)

    for dec in decs:
        _d = data.sel(dec=dec).copy()
        low, hi = map(int, dec.split("-"))
        if monthly:
            from pandas import date_range
            date_range(str(low), periods=len(_d.time), freq='MS')
        else:
            _d.time.values = range(low+1, hi+1)

        dec_dict[dec] = _d

    pol_dict = {}
    pols = data.pol.values.astype(str)

    for pol in pols:
        _p = xr.concat([dec_dict['1980-2010'].sel(pol='REF'),
                        dec_dict['2035-2065'].sel(pol=pol),
                        dec_dict['2085-2115'].sel(pol=pol)],
                       dim='time')
        del _p['pol'], _p['dec']
        _p['pol'] = pol
        _p.set_coords(['pol', ], inplace=True)

        pol_dict[pol] = _p

    merged = xr.auto_combine([pol_dict[pol] for pol in pols], 'pol')

    return merged


def _isin(da, vals):
    """ Determine whether or not values in a given DataArray belong
    to a set of permissible values. """
    return da.to_series().isin(vals).to_xarray()

def poor_isin(arr, vals, op='or'):
    """ This is a hack to check if the values in a given array 'arr' are contained
    in a reference list of values 'vals'. To do this, we simply compute a
    vectorized equality comparison for each element in the list and combine
    them using a bitwise 'or' or 'and', depending on which op is specified by the
    user. A proper "isin" calculation will use the 'or' operator.

    """
    if op not in ['and', 'or']:
        raise ValueError("Unknown op '{}'".format(op))

    mask = np.ones_like(arr) if op == 'and' else np.zeros_like(arr)
    for val in vals:
        if op == 'and':
            mask = mask & (arr == val)
        elif op == 'or':
            mask = mask | (arr == val)
    return mask

class BackToXarray(object):

    def __init__(self, ref_data, mask_field=None):
        self._ref_data = ref_data.copy()
        self.mask_field = mask_field

    @property
    def ref_data(self):
        return self._ref_data

    def back_to_xarray(self, y, name='data', mask_val=None):
        _df = self._ref_data.copy()
        _ds = (
            _df.set_index(['lat', 'lon'])
               .assign(**{name: y, })
               .sortlevel()
               .to_xarray()
        )

        if (mask_val is not None) and (self.mask_field is not None):
            _ds = _ds.where(_ds[self.mask_field] == mask_val)

        return _ds

    def __call__(self, *args, **kwargs):
        return self.back_to_xarray(*args, **kwargs)
