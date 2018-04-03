import ogr
import osr
import os

def export_wkt_multipoints_to_shp(geom, path):
    if os.path.exists(path):
        os.remove(path)

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(path)
    outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbMultiPoint)
    featureDefn = outLayer.GetLayerDefn()

    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(geom)
    outLayer.CreateFeature(outFeature)
    outFeature.Destroy()


def export_coords_to_shp(coordlist, path):
    """Creates a multipoint shapefile of the coordinates in a 2d array, used for debugging purposes.

    :param coordlist: 2d list of coordinates.
    :param path: Output path of the shapefile.
    """
    if os.path.exists(path):
        os.remove(path)

    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    outDataSource = outDriver.CreateDataSource(path)
    outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbMultiPoint)
    featureDefn = outLayer.GetLayerDefn()

    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for plot in coordlist:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(plot[0][0], plot[0][1])
        multipoint.AddGeometry(point)

    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(multipoint)
    outLayer.CreateFeature(outFeature)
    outFeature.Destroy()

def array_to_raster(array, pixel_size, x_min, y_max, wkt, path):
    """Creates a GeoTIFF raster from a numpy array.

    :param array: 2D numpy array of cell values
    :param pixel_size: -- Desired resolution of the output raster, in same units as wkt projection.
    :param x_min: Minimum x coordinate (top left corner of raster)
    :param y_max: Maximum y coordinate
    :param wkt: The wkt string with desired projection
    :param path: The output bath of the GeoTIFF
    """

    import gdal
    import numpy as np
    array = np.rot90(array)
    dst_filename = path
    x_pixels = array.shape[1]  # number of pixels in x
    y_pixels = array.shape[0]  # number of pixels in y
    wkt_projection = wkt

    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1,
        gdal.GDT_Float32, )

    dataset.SetGeoTransform((
        x_min,  # 0
        pixel_size,  # 1
        0,  # 2
        y_max,  # 3
        0,  # 4
        -pixel_size))

    dataset.SetProjection(wkt_projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    return dataset, dataset.GetRasterBand(
        1)  # If you need to return, remenber to return  also the dataset because the band don`t live without dataset.

def utm_lookup(zone):
    """Returns a wkt string of a given UTM zone. Used as a bypass for older las file specifications that do not
    contain wkt strings.


    :param zone: The UTM zone (as a string)
        ex: "10N"
    """
    # see: http://gis.stackexchange.com/questions/233712/python-wkt-or-proj4-lookup-package
    in_ds = ogr.GetDriverByName('CSV').Open('C:\pyformaster\pyformaster\PyFor\pyfor\pcs.csv')
    print(in_ds)
    layer = in_ds.GetLayer()
    layer.SetAttributeFilter("COORD_REF_SYS_NAME LIKE '%UTM zone {}%'".format(zone))
    wkt_string = []
    for feature in layer:
        code = feature.GetField("COORD_REF_SYS_CODE")
        name = feature.GetField("COORD_REF_SYS_NAME")
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(code))
        wkt_string.append(srs.ExportToWkt())
    return ''.join(wkt_string)