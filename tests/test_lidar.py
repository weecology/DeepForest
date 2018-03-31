from DeepForest.Lidar import Lidar

def test_Lidar():
    test_class=Lidar(filename="tests/data/NEON_D03_OSBS_DP1_405000_3276000_classified_point_cloud.laz")
    assert(len(test_class.tile.points)==2912035)
