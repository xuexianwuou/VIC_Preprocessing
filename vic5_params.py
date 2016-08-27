import netCDF4 as netcdf
import numpy as np
from pyproj import Proj
from shapely.geometry import shape


def calc_area(lon, lat, res):
    """Calculate area of polygon after re-projecting to Albers-Equal-Area"""
    co = {"type": "Polygon",
          "coordinates": [[(lon+res/2., lat+res/2.), (lon+res/2., lat-res/2.),
                           (lon-res/2., lat-res/2.), (lon-res/2., lat+res/2.)]]}
    lons, lats = zip(*co["coordinates"][0])
    pa = Proj("+proj=aea +lat_0=%f +lon_0=%f" % (lat, lon))
    x, y = pa(lons, lats)
    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
    return shape(cop).area


def create_area(lats, lons):
    """Create area grid."""
    area = np.zeros((len(lats), len(lons)))
    res = lats[1] - lats[0]
    for i in range(len(lats)):
        for j in range(len(lons)):
            area[i, j] = calc_area(lons[j], lats[i], res)
    return area


def create_mask(soilfile):
    """Create domain mask from VIC4 soil file."""
    soil = np.loadtxt(soilfile)
    lat = soil[:, 2]
    lon = soil[:, 3]
    lons = np.sort(np.unique(lon))
    lats = np.sort(np.unique(lat))
    mask = np.zeros((len(lats), len(lons)), dtype='int')
    res = lats[1] - lats[0]
    idx = np.zeros((len(lat), 3), dtype='int')
    for c in range(len(lat)):
        i = int((lat[c] - lats[0]) / res)
        j = int((lon[c] - lons[0]) / res)
        mask[i, j] = 1
        idx[c, :] = [i, j, soil[c, 1]]
    return mask, lats, lons, idx


def write_domain(soilfile, ncfile):
    """Write domain NetCDF file based on VIC4 soil file."""
    mask, lat, lon, _ = create_mask(soilfile)
    area = create_area(lat, lon)
    fout = netcdf.Dataset(ncfile, 'w')
    fout.createDimension('lat', len(lat))
    fout.createDimension('lon', len(lon))
    maskv = fout.createVariable('mask', 'i4', ('lat', 'lon'))
    areav = fout.createVariable('area', 'f8', ('lat', 'lon'))
    latv = fout.createVariable('lat', 'f8', ('lat'))
    lonv = fout.createVariable('lon', 'f8', ('lon'))
    latv[:] = lat
    lonv[:] = lon
    maskv[:] = mask
    areav[:] = area
    fout.close()


def create_soil(soilfile, nlayer):
    """Generate dictionary of soil data."""
    soil = np.loadtxt(soilfile)
    mask, lats, lons, idx = create_mask(soilfile)
    vars = {'run_cell': [0], 'gridcell': [1], 'cellnum': [1], 'lats': [2], 'lons': [3], 'infilt': [4], 'Ds': [5], 'Dsmax': [6], 'Ws': [7], 'c': [8], 'expt': [9+l for l in range(nlayer)], 'Ksat': [9+nlayer+l for l in range(nlayer)], 'phi_s': [2*nlayer+9+l for l in range(nlayer)], 'init_moist': [3*nlayer+9+l for l in range(nlayer)], 'elev': [4*nlayer+9], 'depth': [4*nlayer+10+l for l in range(nlayer)], 'avg_T': [5*nlayer+10], 'dp': [5*nlayer+11], 'bubble': [5*nlayer+12+l for l in range(nlayer)], 'quartz': [6*nlayer+12+l for l in range(nlayer)], 'bulk_density': [7*nlayer+12+l for l in range(nlayer)], 'soil_density': [8*nlayer+12+l for l in range(nlayer)], 'off_gmt': [9*nlayer+12], 'Wcr_FRACT': [9*nlayer+13+l for l in range(nlayer)], 'Wpwp_FRACT': [10*nlayer+13+l for l in range(nlayer)], 'rough': [11*nlayer+13], 'snow_rough': [11*nlayer+14], 'annual_prec': [11*nlayer+15], 'resid_moist': [11*nlayer+16+l for l in range(nlayer)], 'fs_active': [12*nlayer+16]}
    data = {}
    for name in vars:
        data[name] = np.zeros((len(vars[name]), len(lats), len(lons)))
    for c in range(soil.shape[0]):
        for name in vars:
            data[name][:, idx[c, 0], idx[c, 1]] = soil[c, vars[name]]
    return data, mask, lats, lons


def write_soil(soilfile, ncfile, nlayer=3):
    """Write soil data dictionary to NetCDF file."""
    soil, mask, lat, lon = create_soil(soilfile, nlayer)
    fout = netcdf.Dataset(ncfile, 'w')
    fout.createDimension('lat', len(lat))
    fout.createDimension('lon', len(lon))
    fout.createDimension('nlayer', nlayer)
    ncvars = {}
    for name in soil:
        if soil[name].shape[0] > 1:
            ncvars[name] = fout.createVariable(name, 'f8', ('nlayer', 'lat', 'lon'))
            for l in range(nlayer):
                ncvars[name][l, :, :] = np.ma.masked_array(soil[name][l, :, :], mask == 0)
        else:
            if name in ['run_cell', 'gridcell']:
                ncvars[name] = fout.createVariable(name, 'i4', ('lat', 'lon'))
            else:
                ncvars[name] = fout.createVariable(name, 'f8', ('lat', 'lon'))
            ncvars[name][:] = np.ma.masked_array(soil[name], mask == 0)
    maskv = fout.createVariable('mask', 'i4', ('lat', 'lon'))
    latv = fout.createVariable('lat', 'f8', ('lat',))
    lonv = fout.createVariable('lon', 'f8', ('lon',))
    lyrv = fout.createVariable('nlayer', 'i4', ('nlayer',))
    maskv[:] = mask
    latv[:] = lat
    lonv[:] = lon
    lyrv[:] = range(1, nlayer+1)
    fout.close()


def read_veglib(veglibfile):
    """Read VIC vegetation library."""
    veglib = {}
    with open(veglibfile) as fin:
        for line in fin:
            if not line.startswith("#"):
                tokens = line.split()
                veglib[int(tokens[0])] = {
                    'overstory': int(tokens[1]),
                    'rarc': float(tokens[2]),
                    'rmin': float(tokens[3]),
                    'LAI': [float(t) for t in tokens[4:16]],
                    'albedo': [float(t) for t in tokens[16:28]],
                    'veg_rough': [float(t) for t in tokens[28:40]],
                    'displacement': [float(t) for t in tokens[40:52]],
                    'wind_h': float(tokens[52]),
                    'RGL': float(tokens[53]),
                    'rad_atten': float(tokens[54]),
                    'wind_atten': float(tokens[55]),
                    'trunk_ratio': float(tokens[56]),
                    'veg_descr': " ".join(tokens[57:])
                }
    veglib[len(veglib)+1] = {'overstory': 0, 'rarc': 0.0, 'rmin': 0.0, 'LAI': [0.0]*12, 'albedo': [0.0]*12, 'veg_rough': [0.0]*12, 'displacement': [0.0]*12, 'wind_h': 0.0, 'RGL': 0.0, 'rad_atten': 0.0, 'wind_atten': 0.0, 'trunk_ratio': 0.0, 'veg_descr': "Bare Soil"}
    return veglib


def read_vegparam(vegparamfile, nvegclasses, rootzones):
    """Read VIC vegetation parameter file."""
    vegparam = {}
    vclass = None
    with open(vegparamfile) as fin:
        for line in fin:
            tokens = line.split()
            if len(tokens) == 2:
                cellid = int(tokens[0])
                nveg = int(tokens[1])
                vegparam[cellid] = {}
                vegparam[cellid]['Nveg'] = nveg
                vegparam[cellid]['LAI'] = np.zeros((nvegclasses, 12))
                vegparam[cellid]['Cv'] = np.zeros(nvegclasses)
                vegparam[cellid]['root_depth'] = np.zeros((nvegclasses, rootzones))
                vegparam[cellid]['root_fract'] = np.zeros((nvegclasses, rootzones))
            elif len(tokens) == 12:
                vegparam[cellid]['LAI'][vclass-1, :] = [float(t) for t in tokens]
            else:
                vclass = int(tokens[0])
                vegparam[cellid]['Cv'][vclass-1] = float(tokens[1])
                vegparam[cellid]['root_depth'][vclass-1, :] = [float(t) for t in tokens[2::2]]
                vegparam[cellid]['root_fract'][vclass-1, :] = [float(t) for t in tokens[3::2]]
    return vegparam


def write_veg(veglibfile, vegparamfile, soilfile, ncfile, rootzones=3):
    veglib = read_veglib(veglibfile)
    vegparam = read_vegparam(vegparamfile, len(veglib), rootzones)
    fout = netcdf.Dataset(ncfile, 'r+')
    fout.createDimension('veg_class', len(veglib))
    fout.createDimension('month', 12)
    fout.createDimension('root_zone', rootzones)
    mask, lats, lons, idx = create_mask(soilfile)
    ncvars = {}
    ncvars['month'] = fout.createVariable('month', 'i4', ('month',))
    ncvars['month'][:] = range(1, 13)
    ncvars['root_zone'] = fout.createVariable('root_zone', 'i4', ('root_zone',))
    ncvars['root_zone'][:] = range(1, rootzones+1)
    ncvars['veg_descr'] = fout.createVariable('veg_descr', str, ('veg_class',))
    ncvars['veg_class'] = fout.createVariable('veg_class', 'i4', ('veg_class',))
    for v, k in enumerate(veglib):
        ncvars['veg_descr'][v] = veglib[k]['veg_descr']
        ncvars['veg_class'][v] = k
    for name in ['overstory', 'rarc', 'rmin', 'wind_h', 'RGL', 'rad_atten', 'wind_atten', 'trunk_ratio']:
        ncvars[name] = fout.createVariable(name, 'f8', ('veg_class', 'lat', 'lon'))
        tmp = np.zeros((len(veglib), len(lats), len(lons)))
        for v, k in enumerate(veglib):
            tmp[v, :, :] = veglib[k][name]
            ncvars[name][v, :, :] = np.ma.masked_array(tmp[v, :, :], mask == 0)
    for name in ['albedo', 'veg_rough', 'displacement']:
        ncvars[name] = fout.createVariable(name, 'f8', ('veg_class', 'month', 'lat', 'lon'))
        tmp = np.zeros((len(veglib), 12, len(lats), len(lons)))
        for v, k in enumerate(veglib):
            for m in range(12):
                tmp[v, m, :, :] = veglib[k][name][m]
                ncvars[name][v, m, :, :] = np.ma.masked_array(tmp[v, m, :, :], mask == 0)
    for name in ['root_depth', 'root_fract']:
        ncvars[name] = fout.createVariable(name, 'f8', ('veg_class', 'root_zone', 'lat', 'lon'))
        tmp = np.zeros((len(veglib), rootzones, len(lats), len(lons)))
        for c in range(len(idx)):
            tmp[:, :, idx[c, 0], idx[c, 1]] = vegparam[idx[c, 2]][name]
        for v in range(len(veglib)):
            for r in range(rootzones):
                ncvars[name][v, r, :, :] = np.ma.masked_array(tmp[v, r, :, :], mask == 0)
    ncvars['Nveg'] = fout.createVariable('Nveg', 'i4', ('lat', 'lon'))
    tmp = np.zeros(mask.shape)
    for c in range(len(idx)):
        tmp[idx[c, 0], idx[c, 1]] = vegparam[idx[c, 2]]['Nveg']
    ncvars['Nveg'][:] = np.ma.masked_array(tmp, mask == 0)
    ncvars['Cv'] = fout.createVariable('Cv', 'f8', ('veg_class', 'lat', 'lon'))
    tmp = np.zeros((len(veglib), len(lats), len(lons)))
    for c in range(len(idx)):
        tmp[:, idx[c, 0], idx[c, 1]] = vegparam[idx[c, 2]]['Cv']
    for v in range(len(veglib)):
        ncvars['Cv'][v, :, :] = np.ma.masked_array(tmp[v, :, :], mask == 0)
    ncvars['LAI'] = fout.createVariable('LAI', 'f8', ('veg_class', 'month', 'lat', 'lon'))
    tmp = np.zeros((len(veglib), 12, len(lats), len(lons)))
    for c in range(len(idx)):
        tmp[:, :, idx[c, 0], idx[c, 1]] = vegparam[idx[c, 2]]['LAI']
    for v in range(len(veglib)):
        for m in range(12):
            ncvars['LAI'][v, m, :, :] = np.ma.masked_array(tmp[v, m, :, :], mask == 0)
    fout.close()


def read_snow(snowbandfile):
    """Read VIC elevation bands file."""
    snow = {}
    data = np.loadtxt(snowbandfile)
    nbands = int((data.shape[1] - 1) / 3)
    for c in range(data.shape[0]):
        cellnum = int(data[c, 0])
        snow[cellnum] = {}
        snow[cellnum]['AreaFract'] = data[c, 1:nbands+1]
        snow[cellnum]['elevation'] = data[c, nbands+1:2*nbands+1]
        snow[cellnum]['Pfactor'] = data[c, 2*nbands+1:3*nbands+1]
    return snow, nbands


def write_snow(snowbandfile, soilfile, ncfile):
    """Write snow band information to NetCDF file."""
    snow, nbands = read_snow(snowbandfile)
    fout = netcdf.Dataset(ncfile, 'r+')
    fout.createDimension('snow_band', nbands)
    bandv = fout.createVariable('snow_band', 'i4', ('snow_band',))
    bandv[:] = range(1, nbands+1)
    mask, lats, lons, idx = create_mask(soilfile)
    ncvars = {}
    for name in ['AreaFract', 'elevation', 'Pfactor']:
        ncvars[name] = fout.createVariable(name, 'f8', ('snow_band', 'lat', 'lon'))
        tmp = np.zeros((nbands, len(lats), len(lons)))
        for c in range(len(idx)):
            tmp[:, idx[c, 0], idx[c, 1]] = snow[idx[c, 2]][name]
        for b in range(nbands):
            ncvars[name][b, :, :] = np.ma.masked_array(tmp[b, :, :], mask == 0)
    fout.close()


if __name__ == '__main__':
    write_domain("../california/california.soil", "domain.california.nc")
    write_soil("../california/california.soil", "params.california.nc")
    write_veg("../california/input/vic_veglib.txt", "../california/input/vic.veg.0625.new.cal.adj.can", "../california/california.soil", "params.california.nc")
    write_snow("../california/input/vic.snow.0625.new.cal.adj.can.5bands", "../california/california.soil", "params.california.nc")
