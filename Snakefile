""" (Snake)Makefile for automating data preparation and analyses performed as
part of the time of emergence project. This script mainly generates intermediate datasets that can be ported to run on a local machine, away from the main archive
of data and model output.

"""

from darpy.utilities import copy_attrs
from air_quality.experiments import ECEnsembleResults, FGMEnsembleResults
from collections import namedtuple

#: Field re-mappings: dict -> namedtuple{ec, fgm}
# fm = namedtuple("field_map", ['ec', 'fgm']
# field_map = {}

FIELDS_FGM = ['O3_SRF', 'TREFHT', 'PM25_ANT', 'PRECT', 'U_SRF', 'V_SRF']
FIELDS_EC  = ['IJ_AVG_S__O3_SRF', 'DAO_FLDS__TS', 'PM25_ANT_SRF',
              'DAO_FLDS__PREACC', 'DAO_FLDS__U10M', 'DAO_FLDS__V10M']


def _get_exp(exp_label):
    if exp_label == 'fgm':
        exp = FGMEnsembleResults(cs='30').exp
    elif exp_label == 'ec':
        exp = ECEnsembleResults().exp
    else:
        raise ValueError("Don't know experiment {}".format(exp_label))
    return exp


def _get_exp_files(wildcards):
    exp_label = wildcards['exp']
    exp = _get_exp(exp_label)
    field = wildcards['field']
    fns = [fn for _, fn in exp.walk_files(field)]
    return fns


rule all_masters:
    input:
        expand('data/processed/{case}.air_quality.monthly.nc',
               case=['fgm', 'ec'])

rule case_master:
    output: "data/processed/{case}.air_quality.monthly.nc"
    run:
        from dask.diagnostics import ProgressBar
        import xarray as xr

        case = wildcards['case']
        exp = _get_exp(case)
        if case == 'fgm':
            fields = FIELDS_FGM
        elif case == 'ec':
            fields = FIELDS_EC
        data = []
        for field in fields:
            print(field)
            def _preprocess(ds, *args, **kwargs):
                return ds[[field, ]]
            chunks = {'lon': 48, 'lat': 72}

            data.append(exp.load(field, master=True,
                                 preprocess=_preprocess,
                                 load_kws=dict(chunks=chunks)))
        data = xr.auto_combine(data)

        # Be sure to squeeze out any len-1 dimensions
        data = xr.decode_cf(data.squeeze())

        # Save to disk
        out_fn = output[0]
        print("Writing to " + out_fn)
        with ProgressBar() as pb:
            data.to_netcdf(out_fn)


rule all_seasonal_timeseries:
    """ For all the fields we wish to study and all the seasons, extract
    the timeseries of data averaged over those seasons.

    """
    input:
        expand('data/processed/seasonal_timeseries/fgm.{field}.{season}.nc',
               field=FIELDS_FGM,
               season=['winter', 'summer', 'DJF', 'MAM', 'SON', 'JJA']),
        expand('data/processed/seasonal_timeseries/ec.{field}.{season}.nc',
               field=FIELDS_EC,
               season=['winter', 'summer', 'DJF', 'MAM', 'SON', 'JJA'])

rule seasonal_timeseries_fgm_ec:
    """ Compute winter and summer seasonal average timeseries for
    a given field from the FGM or EC ensemble.

    There are a few different types of "seasons" that could be inferred:
    1. winter/summer: Compute the winter/summer season by looking at the three
       continous warmest or coolest months for each grid cell, and average the
       field of interest over those values.
    2. traditional seasons: Compute means over specific months.
    """
    # It may not be possible to map input files to outputs for this
    # particular rule because not every single case was run.
    output:
        "data/processed/seasonal_timeseries/{case}.{field}.{season}.nc"
    run:
        from air_quality.analysis import seasonal_timeseries
        from dask.diagnostics import ProgressBar
        import xarray as xr

        case = wildcards['case']
        exp = _get_exp(case)
        field = wildcards['field']
        season = wildcards['season']

        # Figure out the temperature field given the experiment:
        if case == 'fgm':
            ts_field = 'TREFHT'
        elif case == 'ec':
            ts_field = 'DAO_FLDS__TS'

        def _preprocess(ds, *args, **kwargs):
            return ds[[field, ]]
        chunks = {'lon': 48, 'lat': 72}

        with ProgressBar() as pb:
            data = exp.load(field, master=True,
                            preprocess=_preprocess,
                            load_kws=dict(chunks=chunks))
            #data = fgm_exp.create_master(field, data)

            if season in ['winter', 'summer']:
                # Load the reference temperature dataset for computing seasons
                TS = exp.load(ts_field, master=True,
                              load_kws=dict(chunks=chunks))
                # Load into memory so we don't have to read files in the future.
                # data = data.persist()
                data['TEMP'] = TS[ts_field]
                # data = data.rename({field: var_lab})
            data = data.load()

        # Be sure to squeeze out any len-1 dimensions
        data = xr.decode_cf(data.squeeze())

        # Compute seasonal timeseries
        if season in ['winter', 'summer']:
            # Use our auxiliary function to figure out summer/winter based on our
            # definition using contiguous warmest/coldest months
            data_seas = seasonal_timeseries(data, field, "TEMP", seas=season)
        elif season in ['DJF', 'JJA', 'SON', 'MAM']:
            import pandas as pd

            # Use a traditional seasonal definition. To be sure we correctly
            # wrap our DJF data around the year, we'll use a Pandas resampling
            # trick - we'll look at quarterly averages, but end the year in November
            # so that December gets lumped with next year's January and Feburay.
            data_seas = data.resample('Q-NOV', 'time', how='mean')

            # Select season and fix time coordinate
            data_seas = data_seas.where(data_seas['time.season'] == season, drop=True)
            data_seas['year'] = data_seas['time.year']

            # Coerce dates to yyyy-01-01
            pd_years = [pd.Timestamp(year, 1, 1) for year in data_seas['year']]
            data_seas['time'].values[:] = pd_years

            data_seas = data_seas.set_coords("year")

        # Get rid of extra TEMP field
        data_seas = data_seas[[field, ]]

        # Save to disk
        out_fn = output[0]
        print("Writing to " + out_fn)
        data_seas.to_netcdf(out_fn)


rule all_annual_cycles:
    input:
        expand("data/processed/annual_cycle/{case}.cycle.nc",
               case=["ec", "fgm"])


rule extract_annual_cycles:
    input:
        "data/processed/{case}.air_quality.monthly.nc"
    output:
        "data/processed/annual_cycle/{case}.cycle.nc"
    run:
        import xarray as xr
        from dask.diagnostics import ProgressBar

        data = xr.open_dataset(input[0], chunks={})

        # Compute cycle by taking monthly means
        data_cycle = data.groupby('time.month').mean(['time'])

        # Save to disk
        out_fn = output[0]
        print("Writing to " + out_fn)
        with ProgressBar() as pb:
            data_cycle.to_netcdf(out_fn)
