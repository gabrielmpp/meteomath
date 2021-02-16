import numpy as np
from numba import jit


@jit(nopython=True)
def fourth_order_derivative(arr: np.ndarray, dim=0):
    """
    2D numpy array with dims [lat, lon]
    :param arr:
    :return:
    """
    # assert isinstance(arr, np.ndarray), 'Input must be numpy array'
    output = np.zeros_like(arr)

    if dim == 0:
        ysize = np.shape(arr)[0]
        for lat_idx in range(2, np.shape(arr - 2)[0]):
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (4 / 3) * (arr[(lat_idx + 1), lon_idx] -
                                                      arr[(lat_idx - 1), lon_idx]) / 2 \
                                           - (1 / 3) * (arr[(lat_idx + 2), lon_idx] -
                                                        arr[(lat_idx - 2), lon_idx]) / 4

        #  First order uncentered derivative for points close to the poles
        for lat_idx in [0, 1]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[(lat_idx + 1), lon_idx] -
                                            arr[lat_idx, lon_idx]) / 2
        for lat_idx in [-1, -2]:
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (arr[lat_idx, lon_idx] -
                                            arr[lat_idx - 1, lon_idx]) / 2
    elif dim == 1:
        xsize = np.shape(arr)[1]
        for lat_idx in range(np.shape(arr)[0]):
            for lon_idx in range(np.shape(arr)[1]):
                output[lat_idx, lon_idx] = (4 / 3) * (arr[lat_idx, (lon_idx + 1) % xsize] -
                                                      arr[lat_idx, (lon_idx - 1) % xsize]) / 2 \
                                           - (1 / 3) * (arr[lat_idx, (lon_idx + 2) % xsize] -
                                                        arr[lat_idx, (lon_idx - 2) % xsize]) / 4

    return output


def derivative_spherical_coords(da, dim=0):
    EARTH_RADIUS = 6371000  # m
    da = da.sortby('latitude')
    da = da.sortby('longitude')
    da = da.transpose('latitude', 'longitude')
    x = da.longitude.copy() * np.pi / 180
    y = da.latitude.copy() * np.pi / 180
    dx = (da.longitude.values[1] - da.longitude.values[0]) * EARTH_RADIUS * np.cos(y)
    dy = (da.latitude.values[1] - da.latitude.values[0]) * EARTH_RADIUS
    deriv = fourth_order_derivative(da.values, dim=dim)
    deriv = da.copy(data=deriv)

    if dim == 0:
        deriv = deriv / dy
    elif dim == 1:
        deriv = deriv / dx
    else:
        raise ValueError('Dim must be either 0 or 1.')
    return da.copy(data=deriv)
