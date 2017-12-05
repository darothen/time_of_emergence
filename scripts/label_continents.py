#!/usr/bin/env python
""" Given a gridded dataset, compute continent labels for each
grid cell in that dataset and save to disk.

"""
import numpy as np
import xarray as xr

import shapely
from shapely.geometry import asPoint
import geopandas as gpd

from darpy import shift_lons, append_history

from argparse import ArgumentParser, RawDescriptionHelpFormatter
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("dataset", help="Reference dataset with grid to map")
parser.add_argument("output", default="continents.nc",
                    help="Output file containing continent labels")


def label_continents(in_fn, out_fn):
    # Load in a low-res dataset with continental boundaries
    # and dissolve borders.
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    continents = world[['continent', 'geometry']]
    continents = continents.dissolve(by='continent')
    continents = continents.reset_index()

    # Load in a sample dataset with the lat-lon mesh you want
    # to map labels to
    data = xr.open_dataset(in_fn)
    # For CESM data, fix lons to monotonically increase from
    # [-180, 180]
    data = shift_lons(data).roll(lon=len(data.lon)//2 - 1)

    # Assume grid gives cell centers; compute Point correspond to each
    # cell center
    llon, llat = np.meshgrid(data.lon.values, data.lat.values)
    lon_lin, lat_lin = llon.ravel(), llat.ravel()
    points = [asPoint(p) for p in np.column_stack([lon_lin, lat_lin])]

    # Coerce to a geopandas data structure
    pts = gpd.GeoDataFrame({'lon': lon_lin, 'lat': lat_lin, 'geometry': gpd.GeoSeries(points)})
    pts.head()

    # Spatial join to map cells to continent labels
    joined = gpd.sjoin(pts, continents, how='inner', op='intersects')
    joined.head()

    # Re-structure by merging back into original dataset and save to disk
    labels = (
        joined[['lat', 'lon', 'continent']]
        .assign(continent=lambda x: x.continent.astype('category'))
        .set_index(['lat', 'lon'])
        .sortlevel()
        .assign(continent_id=lambda x: x.continent.cat.codes)
        .to_xarray()
    )

    x = (
        data
        .copy()
        .merge(labels)
    )

    to_save = x[['continent', 'continent_id']]
    to_save = append_history(to_save)

    to_save.to_netcdf(out_fn, mode='w')


if __name__ == "__main__":

    args = parser.parse_args()
    label_continents(args.dataset, args.output)
