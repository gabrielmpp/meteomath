import xarray as xr
import numpy as np

def discrete_vertical_integral(zonal_flux, meridional_flux, f, coord = 'pressure'):
    '''
    Method to compute the discrete integral of an array over a given coordinateself.

    Keyword arguments:
    zonal_flux -- xarray DataArray zonal wind speed or flux
    meridional_flux -- xarray DataArray meridional wind speed or flux
    f -- function to apply before the vertical integration. Examples: meteomath.divergence or meteomath.vorticity
    coord - coordinate to integrate upon
    '''

    if not isinstance(zonal_flux, xr.DataArray) or isinstance(meridional_flux, xr.DataArray):
        raise ValueError('The inputs should be xarray dataarray')

    if zonal_flux[vertical_coord] != meridional_flux[vertical_coord]:
        raise ValueError('The vertical coordinates should match')

    vertical_deltas = zonal_flux[vertical_coord].diff(coord)
    div = f(zonal_flux, meridional_flux)
    div = div.interp({vertical_coord: pressure_deltas[coord]})
    div = div*vertical_deltas
    div_int = div.sum(coord)
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
