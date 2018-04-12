# script to update the OSBS ITC data
rm(list=ls(all=TRUE))
overJac <- function(itc, itcSeg){
  tryCatch({ 
    itcSeg <- crop(itcSeg, extent(itc))
    #check if the polygon is empty: put a warning and leave the function
    
    #itcIntercept <- over(itc, itcSeg)
    overITC <- rep(NA, length(itc))
    
    for(iii in 1:length(overITC)){
      foo <- itc[iii,]
      if(!is.null(itcSeg)){
        #plot(itc)
        #plot(itcSeg, axes=T, border="blue", add=TRUE, lwd = 1.5)
        x1 <- over(itcSeg, foo, returnList=TRUE)
        #plot(itcSeg[iii,], add=T, border = 'red')
        x1_mask <- rep(0, length(x1))
        for(jjj in 1:length(x1)){
          x1_mask[jjj] <- gIntersects(foo, itcSeg[jjj,])
        }
        x1_mask
        itcSubSeg <- itcSeg[x1_mask==1,]
        
        #plot(itcSubSeg)
        #plot(foo, axes=T, border="blue", add=TRUE, lwd = 1.5)
        if(sum(x1_mask) == 0){
          bOverlap = 0
        }else{
          bOverlap <- rep(NA, length(x1))
          for(mmm in 1:length(itcSubSeg)){
            bOverlap[mmm] <- gArea(gIntersection(foo, itcSubSeg[mmm,]))/
              (gArea(foo) + gArea(itcSubSeg[mmm,]) - gArea(gIntersection(foo, itcSubSeg[mmm,])))
          }
          overITC[iii] <- max(bOverlap,  na.rm = T)
        }
      }else{
        overITC <- 0
      }  
    }
    aveOve <- overITC #(mean(overITC, na.rm = T))
  }, error=function(e){cat(paste("ERROR in ", itc, " :" ,sep="~"),conditionMessage(e), "\n")})
  return(aveOve)
}


## Define the function
gdal_polygonizeR <- function(x, outshape=NULL, gdalformat = 'ESRI Shapefile',
                             pypath=NULL, readpoly=TRUE, quiet=TRUE) {
  if (isTRUE(readpoly)) require(rgdal)
  if (is.null(pypath)) {
    pypath <- Sys.which('gdal_polygonize.py')
  }
  if (!file.exists(pypath)) stop("Can't find gdal_polygonize.py on your system.")
  owd <- getwd()
  on.exit(setwd(owd))
  setwd(dirname(pypath))
  if (!is.null(outshape)) {
    outshape <- sub('\\.shp$', '', outshape)
    f.exists <- file.exists(paste(outshape, c('shp', 'shx', 'dbf'), sep='.'))
    if (any(f.exists))
      stop(sprintf('File already exists: %s',
                   toString(paste(outshape, c('shp', 'shx', 'dbf'),
                                  sep='.')[f.exists])), call.=FALSE)
  } else outshape <- tempfile()
  if (is(x, 'Raster')) {
    require(raster)
    writeRaster(x, {f <- tempfile(fileext='.tif')})
    rastpath <- normalizePath(f)
  } else if (is.character(x)) {
    rastpath <- normalizePath(x)
  } else stop('x must be a file path (character string), or a Raster object.')
  system2('python', args=(sprintf('"%1$s" "%2$s" -f "%3$s" "%4$s.shp"',
                                  pypath, rastpath, gdalformat, outshape)))
  if (isTRUE(readpoly)) {
    shp <- readOGR(dirname(outshape), layer = basename(outshape), verbose=!quiet)
    return(shp)
  }
  return(NULL)
}

library(rgdal)
library(dplyr)
library(raster)
library(itcSegment)
library(doParallel)
library(rlas)

#### READ IN DATA
tagged_spdf <- readOGR(dsn="./NIST_data_20170317", layer = "Final Tagged Trees",stringsAsFactors = F)
untagged_spdf <- readOGR(dsn="./NIST_data_20170317", layer = "Final Untagged Trees", stringsAsFactors = F)

tag_df <- tagged_spdf@data
utag_df <- untagged_spdf@data
tag_df$id <- as.integer(tag_df$id)

# how many crowns in total? 628
dim(tag_df)[1] + dim(utag_df)[1]
#listPlot <- as.vector(read.csv('plotsList.csv', header = F, stringsAsFactors = F))$V1

listPlot <- list.files("./vector/", pattern = glob2rx("ITC*.shp"))
listPlot <- substr(listPlot,1,nchar(listPlot)-4) 
in.dir = "./vector"
epsg <- 32617
#jac.dalponte = jac.silva = jac.water = jac.dlp = rep(NA, length(listPlot))
jac.dalponte = jac.silva = jac.water = jac.dlp = list()
for(i in listPlot){
  tryCatch({ 
    
    library(raster)
    library(lidR)

    print(i)
    itc <- readOGR(dsn=in.dir, layer = i,stringsAsFactors = F)
    itc <-spTransform(itc, CRS("+init=epsg:32617"))
    extnt <- extent(itc)
    path.raster <- "./RSdata/"
    path.pointCloud <- "./Classified_point_cloud"
    pt = list.files(path.pointCloud, pattern=paste(as.integer(extnt@xmin/1000)*1000, 
                                                   as.integer(extnt@ymax/1000)*1000, sep="_"))
    
    las = readLAS(paste(path.pointCloud, pt, sep="/"))
    
    lasnormalize(las, method = "knnidw", k = 10L)
    #lasnormalize(las, method = "kriging")
    
    # compute a canopy image
    chm = grid_canopy(las, res = 0.5, subcircle = 0.2, na.fill = "knnidw", k = 4)
    chm = as.raster(chm)
    kernel = matrix(1,3,3)
    chm = raster::focal(chm, w = kernel, fun = mean)
    chm = raster::focal(chm, w = kernel, fun = mean)
    
    #crop chm on raster
    chm.crop = crop(x = chm, y = extnt)
    #plot(chm.crop, col = gray.colors(10, start = 0.3, end = 0.9, gamma = 2.2, alpha = NULL))
    #plot(itc, add=T, border = "black")
    
    #useITCsegment to extract ITCs
    itcSeg.dalponte <- itcIMG(chm.crop, epsg = 32617) 
    #plot(itcSeg.dalponte, add=T, border = "red")
    
    #silva 2016
    ttops = tree_detection(chm.crop, 5, 2)
    crowns <-lastrees_silva(las, chm.crop, ttops, max_cr_factor = 0.6, exclusion = 0.3, extra = T)
    itcSeg.silva <- gdal_polygonizeR(crowns)
    proj4string(itcSeg.silva) <-  CRS(paste("+init=epsg:", epsg, sep=""))
    #plot(itcSeg.silva, add = T, border = "blue")
    
    # # tree segmentation
    # Watershed
    crowns <-lastrees_watershed(las, chm.crop, th_tree = 5,extra = T)
    itcSeg.wtsh <- gdal_polygonizeR(crowns)
    proj4string(itcSeg.wtsh) <-  CRS(paste("+init=epsg:", epsg, sep=""))
    
    #plot(itcSeg.wtsh, add = T, border = "brown")
    
    # 
    # #dalponte
    ttops = tree_detection(chm.crop, 5, 2)
    crowns <- lastrees_dalponte(las, chm.crop, ttops, max_cr = 7, extra = T)
    itcSeg.dlp <- gdal_polygonizeR(crowns)
    proj4string(itcSeg.dlp) <-  CRS(paste("+init=epsg:", epsg, sep=""))
    
    #plot(itcSeg.dlp, add = T, border = "purple")
    
    jac.dalponte[[i]] <- overJac(itc, itcSeg.dalponte)
    jac.silva[[i]] <- overJac(itc, itcSeg.silva)
    jac.water[[i]] <- overJac(itc, itcSeg.wtsh)
    jac.dlp[[i]] <- overJac(itc, itcSeg.dlp)
    
    
  }, error=function(e){cat(paste("ERROR in ", i, " :" ,sep="~"),conditionMessage(e), "\n")})
  
}

final.dalponte <- mean(unlist(jac.dalponte), na.rm = T)
final.silva <- mean(unlist(jac.silva), na.rm = T)
final.water <- mean(unlist(jac.water), na.rm = T)
final.dlp <- mean(unlist(jac.dlp), na.rm = T)
hist(unlist(jac.silva))
summary(unlist(jac.dalponte), na.rm = T)
summary(unlist(jac.silva), na.rm = T)
summary(unlist(jac.water), na.rm = T)
summary(unlist(jac.dlp), na.rm = T)

