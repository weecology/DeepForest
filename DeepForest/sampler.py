import ogr
import numpy as np

class PlotSampler:
    """Handles and creates information from a CloudInfo object for sampling.
    Requres a CloudInfo object."""

    def __init__(self, cloud):

        self.cloud = cloud

        self.plot_shp = None
        self.plot_cells = None
        self.vert_list = None

        # Check if grid_constructor has been run.

        if cloud.grid_x == None or cloud.grid_y == None:
            print("At least one grid dimension is missing, consider running the grid_constructor function.")

    def export_grid(self, path, verts=False):
        """Creates a grid shapefile from the cloud grid_constructor function."""
        import numpy as np
        import os
        """Exports grid to shapefile for further analysis."""
        # TODO: Consider moving into PointCloud?

        print("Exporting grid to .shp.")

        self.grid_path = path

        def list_append(n, list1, start="end"):
            step = list1[1] - list1[0]
            if start == "end":
                last = list1[-1]
                for i in range(n):
                    list1.append(last + (i + 1) * step)
                return list1
            if start == "beginning":
                first = list1[0]
                for i in range(n):
                    list1.insert(0, first - (i + 1) * step)
                return list1

        xs = list_append(3, self.cloud.grid_x)
        ys = list_append(3, self.cloud.grid_y, start="beginning")

        mesh = np.meshgrid(xs, ys)

        def vertices(origin_x, origin_y):
            """Returns a list of the vertices of a grid cell at origin_x and origin_y"""
            try:
                top_left = (float(mesh[0][origin_y][origin_x]), float(mesh[1][origin_y][origin_x]))
                top_right = (float(mesh[0][origin_y][origin_x + 1]), float(mesh[1][origin_y][origin_x + 1]))
                bottom_left = (float(mesh[0][origin_y + 1][origin_x]), float(mesh[1][origin_y + 1][origin_x]))
                bottom_right = (float(mesh[0][origin_y + 1][origin_x + 1]), float(mesh[1][origin_y + 1][origin_x + 1]))
                return [top_left, top_right, bottom_right, bottom_left]
            except IndexError:
                pass

        def vert_list(xs, ys):
            """Compiles the vertices of a grid into a list of square vertices for each grid cell."""
            a = 0
            b = 0
            verts = []
            for row in ys:
                for col in xs:
                    if vertices(a, b) != None:
                        verts.append(vertices(a, b))
                    else:
                        pass
                    a += 1
                a = 0
                b += 1
            return verts

        if os.path.exists(path):
            os.remove(path)

        outDriver = ogr.GetDriverByName('ESRI Shapefile')
        outDataSource = outDriver.CreateDataSource(path)
        outLayer = outDataSource.CreateLayer(path, geom_type=ogr.wkbPolygon)
        featureDefn = outLayer.GetLayerDefn()

        self.vert_list = vert_list(xs,ys)
        for vert_set in self.vert_list:
            ring = ogr.Geometry(ogr.wkbLinearRing)
            first_point = vert_set[0]
            for point in vert_set:
                ring.AddPoint(point[0], point[1])
            ring.AddPoint(first_point[0], first_point[1])
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)

            outLayer.CreateFeature(outFeature)
            outFeature.Destroy()
        if verts ==True:
            import gisexport
            gisexport.export_coords_to_shp(self.vert_list, "gridverts.shp")


    def get_geom(self, path):
        """Reads a shapefile and returns a list of Wkt geometries.

        Keyword arguments:
        path -- Input path of a shapefile.
        """
        # TODO: Put in gisexport?
        geoms = []
        shapefile = path
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(shapefile, 0)
        layer = dataSource.GetLayer()

        for feature in layer:
            geom = feature.GetGeometryRef()
            geoms.append(geom.ExportToWkt())
        return geoms


    def intersect_plots(self):
        """Exports a list of JSON coordinates of cell vertices near the plot location. Used to clip a square around
        the plots within a las file."""

        plot_shp = self.plot_shp
        plot_geoms = self.get_geom(plot_shp)
        print(plot_geoms[0])
        self.all_cells = []
        if plot_shp == None:
            print("Consider adding a plot shapefile path.")
        else:
            print("Indexing plots with grid.")
            for plot in plot_geoms:
                plot_cells = []
                plot = ogr.CreateGeometryFromWkt(plot)
                grid = self.get_geom(self.grid_path)
                for cell in grid:
                    cell = ogr.CreateGeometryFromWkt(cell)
                    if cell.Overlaps(plot) == True or cell.Within(plot) == True:
                        json_cell = cell.ExportToJson()
                        plot_cells.append(json_cell)
                self.all_cells.append(plot_cells)

    def extract_points(self):
        """Returns a list of plot coordinate tuples (len = plots)"""
        import json
        import itertools

        self.intersect_plots()

        geoms = self.all_cells # one geom per plot

        all_points = []

        for plot in geoms: # for each plot
            coords = []
            for json_geom in plot: # for each json geom
                json_in = json.loads(json_geom)# load the json
                coords += [coord_pair for coord_pair in json_in['coordinates'][0]] # compile coord for each dict into list of tuples
            coords.sort()
            unique_points = list(coords for coords,_ in itertools.groupby(coords)) # only unique verts for the plot are in this list
            all_points.append(unique_points)
        return all_points

    def df_sort(self, plot_set):
        """Takes all of the unique vertices and makes a new dataframe"""
        #FIXME: Does not work if plot hangs over extent (and it probably shouldn't)
        import pandas as pd

        cloud_df = self.cloud.dataframe
        cloud_group = cloud_df.groupby(['cell_x', 'cell_y'])

        plot_set = [tuple(i) for i in plot_set]

        frames = []

        for coord in plot_set:
            frames.append(cloud_group.get_group(coord))

        newdf = pd.concat(frames)

        return newdf

    def extract_plot(self, square_df, plot_index):
        """converts dataframe points to something"""

        plot = self.get_geom(self.plot_shp)[plot_index]
        plot = ogr.CreateGeometryFromWkt(plot)

        plot_df = square_df

        point_array = np.column_stack([plot_df.index.values, plot_df.x.values, plot_df.y.values])

        plot_points = []
        for point in point_array:
            thing = ogr.Geometry(ogr.wkbPoint)
            thing.AddPoint(point[1], point[2])
            if thing.Within(plot):
                plot_points.append(point[0])

        new_df = plot_df.loc[plot_points, :]
        return new_df

    def clip_plots(self):
        """Creates a set of clipped las files based on the input plot shapefile."""
        #TODO: Potential for quadtree here!
        from PyFor.pyfor import normalize
        header = self.cloud.header
        unique_plot_points = self.extract_points()
        i=1
        for point_set in unique_plot_points:
            print("Writing a plot to .las", i)
            #FIXME: This gives weird file names.
            filename = "plot" + str(i) + ".las"
            file_path = filename
            normalize.df_to_las(self.extract_plot(self.df_sort(point_set), i-1),
                                file_path, header, 'norm')
            i+=1