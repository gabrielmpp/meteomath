import xarray as xr
import numpy as np

def vertical_div_from_flux(zonal_flux, meridional_flux, vertical_coord = 'pressure'):

    vertical_deltas = zonal_flux[vertical_coord].diff(vertical_coord)
    div = divergence(zonal_flux, meridional_flux)
    div = div.interp({vertical_coord: pressure_deltas[vertical_coord]})
    div = div*vertical_deltas
    div_int = div.sum(vertical_coord)
    return div_int

def to_cartesian(array, lon_name = 'longitude', lat_name = 'latitude', earth_r = 6371000):

    array['x'] = array[lon_name]*np.pi*earth_r/180
    array['y'] = xr.apply_ufunc(lambda x: np.sin(np.pi*x/180)*earth_r, array[lat_name])
    return array

def divergence(u, v):

    return u.differentiate('x') + v.differentiate('y')

def vorticity(u, v):

    return (v.differentiate('x') - u.differentiate('y'))

def strain_rate(u, v):

    return 0.5*(u.differentiate('y') + v.differentiate('x'))
