import pandas as pd
import numpy as np

def elev_points(tiff, cloud):
    """Normalizes a point cloud.

    :param tiff: A GeoTIFF with extent that covers all considered LiDAR points, must match projections.
    :param cloud: A CloudInfo object.
    """
    #TODO: Match tiff and cloud projections.
    import gdal
    import affine

    print("Normalizing points.")

    xy_array = cloud.dataframe.as_matrix(columns=['x', 'y'])

    # Retrieving tiff information & transformation.
    raster_object = gdal.Open(tiff)
    raster_array = np.array(raster_object.GetRasterBand(1).ReadAsArray())
    geo_trans = raster_object.GetGeoTransform()
    forward_transform = affine.Affine.from_gdal(*geo_trans)
    reverse_transform = ~forward_transform

    def retrieve_pixel_value(coord_array):
        """Returns an array of pixel values underneath each LiDAR point.

        :param coord_array: A 2D numpy array of XY coordinates.
        """
        x_coords, y_coords = xy_array[:, 0], xy_array[:, 1]
        pixel_x, pixel_y = reverse_transform * (x_coords, y_coords)
        pixel_x, pixel_y = pixel_x + 0.5, pixel_y + 0.5
        pixel_x, pixel_y = pixel_x.astype(int), pixel_y.astype(int)
        return raster_array[pixel_y, pixel_x]


    cloud.dataframe['elev'] = retrieve_pixel_value(xy_array)
    cloud.dataframe['norm'] = cloud.dataframe['z'] - cloud.dataframe['elev']

    # Some cleaning processes.
    cloud.dataframe.dropna(inplace=True)
    cloud.dataframe = cloud.dataframe[cloud.dataframe.elev != 0]


def df_to_las(df, out_path, header, z_col = 'norm'):
    """Exports normalized points to new las.

    :param df: A dataframe of las information to write from.
    :param out_path: The location and name of the output file.
    :param header: The header object to write to the output file.
    :param zcol: The elevation (z) information to be written to the file.
    """
    print("Writing dataframe to .las")

    import laspy

    outfile = laspy.file.File(out_path, mode="w", header = header)
    outfile.x = df['x']
    outfile.y = df['y']
    outfile.z = df[z_col]
    outfile.intensity = df['int']
    #TODO: Fix, currently not working
    #outfile.return_num = df['ret']

def ground_to_las(df, out_path, header):
    # #TODO: Merger with df_to_las
    #Get all ground classifcations in a new dataframe.

    df = df.loc[df['classification'] == 2]


    import laspy

    outfile = laspy.file.File(out_path, mode = "w", header = header)

    outfile.x = df['x']
    outfile.y = df['y']
    outfile.z = df['z']

