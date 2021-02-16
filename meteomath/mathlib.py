"""
Set of functions to apply common mathematical operations on xarray DataArrays
"""

import xarray as xr
import numpy as np
from typing import Tuple
from meteomath.finite_differences import derivative_spherical_coords

def interpolate_c_stagger(
        u: xr.DataArray,
        v: xr.DataArray,
        h: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Function to interpolate regular xr.DataArrays onto the Arakawa C stagger.

    :param u: zonal wind
    :param v: meridional wind
    :return: u, v interpolated in the Arakawa C stagger and h_grid in the midpoints
    """
    regular_u, delta_lat_u, delta_lon_u = _assert_regular_latlon_grid(u)
    regular_v, delta_lat_v, delta_lon_v = _assert_regular_latlon_grid(v)
    regular_h, delta_lat_h, delta_lon_h = _assert_regular_latlon_grid(h)
    assert regular_u, 'u array lat lon grid is not regular'
    assert regular_v, 'v array lat lon grid is not regular'
    assert regular_h, 'h array lat lon grid is not regular'
    assert delta_lat_u == delta_lat_v, 'u and v lat are not compatible'
    assert delta_lon_u == delta_lon_v, ' u and v lon are not compatible'
    u_new = u.copy(data=np.zeros(u.shape))
    v_new = v.copy(data=np.zeros(v.shape))
    u_new = u_new.assign_coords(latitude=u.latitude - delta_lat_u*0.5)
    v_new = v_new.assign_coords(longitude=v.longitude - delta_lon_v*0.5)
    u_new = u_new.isel(latitude=slice(None, -1))
    v_new = v_new.isel(longitude=slice(None, -1))
    u_new = u.interp_like(u_new)
    v_new = v.interp_like(v_new)
    h_grid = xr.DataArray(0, dims=['latitude', 'longitude'], coords=[
        u_new.latitude, v_new.longitude
    ])
    h_new = h.interp_like(h)

    return u_new, v_new, h_new


def _assert_regular_latlon_grid(array: xr.DataArray) -> Tuple[bool, np.array, np.array]:
    """
    Method to assert if an array latitude and longitude dimensions are regular.
    :param array: xr.DataArray to be asserted
    :return: Tuple[bool, np.array, np.array]
    """
    delta_lat = (array.latitude.shift(latitude=1) - array.latitude).dropna('latitude').values
    delta_lat = np.unique(np.round(delta_lat, 5)) # TODO rounding because numpy unique seems to have truncation error
    delta_lon = (array.longitude.shift(longitude=1) - array.longitude).dropna('longitude').values
    delta_lon = np.unique(np.round(delta_lon, 5))
    if delta_lat.shape[0] == 1 and delta_lon.shape[0] == 1:
        regular = True
    else:
        regular = False
    return regular, delta_lat, delta_lon


def discrete_vertical_integral(zonal_flux, meridional_flux, f, coord='pressure'):
    """
    Method to compute the discrete integral of an array over a given coordinate.

    :param zonal_flux: xarray DataArray zonal wind speed or flux
    :param meridional_flux: xarray DataArray meridional wind speed or flux
    :param f: function to apply before the vertical integration. Examples: meteomath.divergence or meteomath.vorticity
    :param coord: coordinate to integrate upo
    :return:
    """

    assert (isinstance(zonal_flux, xr.DataArray) and isinstance(meridional_flux, xr.DataArray)), \
        'The inputs should be xarray dataarray'

    #if zonal_flux[coord] != meridional_flux[coord]:
    #    raise ValueError('The vertical coordinates should match')

    vertical_deltas = zonal_flux[coord].diff(coord)
    div = f(zonal_flux, meridional_flux)
    div = div.interp({coord: vertical_deltas[coord]})
    div = div*vertical_deltas
    div_int = div.sum(coord)
    return div_int


def to_cartesian(array, lon_name='longitude', lat_name='latitude', earth_r=6371000):
    """
    Method to include cartesian coordinates in a lat lon xr. DataArray

    :param array: input xr.DataArray
    :param lon_name: name of the longitude dimension in the array
    :param lat_name: name of the latitude dimension in the array
    :param earth_r: earth radius
    :return: xr.DataArray with x and y cartesian coordinates
    """
    array['x'] = array[lon_name]*np.pi*earth_r/180
    array['y'] = xr.apply_ufunc(lambda x: np.sin(np.pi*x/180)*earth_r, array[lat_name])
    return array


def divergence(u, v):
    """
    Method to compute the 2D divergence of u and v arrays on cartesian coordinates.
    :param u: xr.DataArray
    :param v: xr.DataArray
    :return: xr.DataArray
    """

    return derivative_spherical_coords(u, dim=1) + derivative_spherical_coords(v, dim=0)


def vorticity(u, v):
    """
    Method to compute the 2D vorticity of u and v arrays on cartesian coordinates.
    :param u: xr.DataArray
    :param v: xr.DataArray
    :return: xr.DataArray
    """
    return derivative_spherical_coords(v, dim=1) - derivative_spherical_coords(u, dim=0)


def strain_rate(u, v):
    """
    Method to compute the 2D strain rate of u and v arrays on cartesian coordinates.
    :param u: xr.DataArray
    :param v: xr.DataArray
    :return: xr.DataArray
    """
    return 0.5*(derivative_spherical_coords(u, dim=0) + derivative_spherical_coords(v, dim=1))