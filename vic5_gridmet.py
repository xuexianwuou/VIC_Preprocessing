import netCDF4 as netcdf
import numpy as np
from datetime import datetime, timedelta
from vic5_params import create_mask


def write_forcing(filename, year, prec, tmax, tmin, wind, rlat, rlon, mask):
    """Write NetCDF VIC forcing file."""
    fout = netcdf.Dataset(filename, 'w')
    fout.createDimension('lat', len(rlat))
    fout.createDimension('lon', len(rlon))
    fout.createDimension('time', prec.shape[0])
    latv = fout.createVariable('lat', 'f8', ('lat',))
    latv[:] = rlat
    latv.units = "degrees_north"
    lonv = fout.createVariable('lon', 'f8', ('lon',))
    lonv[:] = rlon
    lonv.units = "degrees_east"
    tv = fout.createVariable('time', 'i4', ('time',))
    tv.units = "hours since {0}-01-01 00:00:00".format(year)
    tv.calendar = "proleptic_gregorian"
    tv[:] = netcdf.date2num([datetime(year, 1, 1) + timedelta(days=t) for t in range(prec.shape[0])], units=tv.units, calendar=tv.calendar)
    precv = fout.createVariable('prcp', 'f8', ('time', 'lat', 'lon'))
    precv.units = "mm"
    tmaxv = fout.createVariable('tmax', 'f8', ('time', 'lat', 'lon'))
    tmaxv.units = "K"
    tminv = fout.createVariable('tmin', 'f8', ('time', 'lat', 'lon'))
    tminv.units = "K"
    windv = fout.createVariable('wind', 'f8', ('time', 'lat', 'lon'))
    windv.units = "m/s"
    for t in range(prec.shape[0]):
        precv[t, :, :] = np.ma.masked_array(prec[t, :, :], mask == 0)
        tmaxv[t, :, :] = np.ma.masked_array(tmax[t, :, :], mask == 0)
        tminv[t, :, :] = np.ma.masked_array(tmin[t, :, :], mask == 0)
        windv[t, :, :] = np.ma.masked_array(wind[t, :, :], mask == 0)
    maskv = fout.createVariable('mask', 'i4', ('lat', 'lon'))
    maskv[:] = mask
    fout.close()


def process_file(filepath, varname, year, mask, rlat, rlon):
    """Process GRIDMET NetCDF file and subset to provided mask."""
    var = {'pr': 'precipitation_amount', 'vs': 'wind_speed', 'tmmx': 'air_temperature', 'tmmn': 'air_temperature'}
    ncfilename = "{0}/{1}_{2}.nc".format(filepath, varname, year)
    ds = netcdf.Dataset(ncfilename)
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    i = np.where(np.logical_and(lat >= min(rlat)-2., lat <= max(rlat)+2.))[0]
    j = np.where(np.logical_and(lon >= min(rlon)-2., lon <= max(rlon)+2.))[0]
    data = ds.variables[var[varname]][:, i, j]
    lat = lat[i]
    lon = lon[j]
    rdata = np.zeros((data.shape[0], len(rlat), len(rlon)))
    # FIXME: Uses nearest-neighbor interpolation, should probably change to something more sophisticated
    for ri in range(len(rlat)):
        for rj in range(len(rlon)):
            if mask[ri, rj] == 1:
                i = np.argmin(abs(rlat[ri] - lat))
                j = np.argmin(abs(rlon[rj] - lon))
                rdata[:, ri, rj] = data[:, i, j]
    return rdata


def forcings(soilfile, filepath, ncfilename, year):
    """Generate VIC5 forcings."""
    mask, rlat, rlon, _ = create_mask(soilfile)
    prec = process_file(filepath, "pr", year, mask, rlat, rlon)
    tmax = process_file(filepath, "tmmx", year, mask, rlat, rlon)
    tmin = process_file(filepath, "tmmn", year, mask, rlat, rlon)
    wind = process_file(filepath, "vs", year, mask, rlat, rlon)
    write_forcing("{0}.{1}.nc".format(ncfilename, year), year, prec, tmax, tmin, wind, rlat, rlon, mask)


if __name__ == '__main__':
    for year in range(1979, 2016):
        forcings("../california/california.soil", "/Volumes/External2/gridmet", "california_forcing", year)
        print("Finished processing year {}".format(year))
