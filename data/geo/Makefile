
# Define some grids to help with CDO remapping
elev.1-deg.grid: elev.1-deg.nc
	cdo griddes $< > $@
fgm.grid: ../processed/REF.CS30.IC1.1980-2010.TS.monthly.nc
	cdo griddes $< > $@

# Re-map 1 degree elevation to FGM grid
elev.fgm-grid.nc: elev.1-deg.nc fgm.grid
	cdo remapbil,fgm.grid elev.1-deg.nc elev.fgm-grid.nc
	ncks -A -v lat ../processed/REF.CS30.IC1.1980-2010.TS.monthly.nc elev.fgm-grid.nc

# Label continents on grid
cont.fgm-grid.nc: elev.fgm-grid.nc
	python ../../scripts/label_continents.py $< $@
