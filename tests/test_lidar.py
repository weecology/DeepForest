def test_Lidar():
    test_class=Lidar(filename="data/NEON_D03_OSBS_DP1_405000_3276000_classified_point_cloud.laz")
    assert(test_class.filename=="data/NEON_D03_OSBS_DP1_405000_3276000_classified_point_cloud.laz")