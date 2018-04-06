segment_ITC<-function(las,algorithm="watershed"){
  
  ws = seq(3,21, 3)
  th = seq(0.1, 2, length.out = length(ws))
  
  lasground(las, "pmf", ws, th)
  
  # normalization
  lasnormalize(las, method = "knnidw", k = 10L)
  
  # compute a canopy image
  chm= grid_canopy (las, 1, subcircle = 0.1, na.fill = "knnidw", k = 3, p = 2)
  chm = as.raster(chm)
  kernel = matrix(1,3,3)
  chm = raster::focal(chm, w = kernel, fun = mean)
  chm = raster::focal(chm, w = kernel, fun = mean)
  
  # tree segmentation
  crowns = lastrees(las, algorithm = algorithm, chm, th = 4, extra = TRUE)
  
  # display
  tree = lasfilter(las, !is.na(treeID))
  plot(tree, color = "treeID", colorPalette = pastel.colors(100), size = 1)
  
  # More stuff
  contour = rasterToPolygons(crowns, dissolve = TRUE)
  
  plot(chm, col = height.colors(50))
  plot(contour, add = T)
  return(contour)
}