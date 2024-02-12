## Detection network training
library(TreeSegmentation)
library(stringr)
library(dplyr)

#Location of the training tiles
save_dir="/blue/ewhite/DeepForest/pretraining"
lidar_tiles<-read.csv("lidar_tiles.csv")[,2]
rgb_tiles<-read.csv("rgb_tiles.csv")[,2]
rgb_tiles<-data.frame(RGB=rgb_tiles) %>% filter(str_detect(RGB,"Mosaic")) %>% filter(str_detect(RGB,"image"))
rgb_tiles$geo_index<-str_match(rgb_tiles$RGB,"(\\d+_\\d+)_image")[,2]
rgb_tiles$year<-str_match(rgb_tiles$RGB,"DP3.30010.001/neon-aop-products/(\\d+)/FullSite/")[,2]

#Set parameters based on whether site is decidous ("D") or coniferous ("C")
site_df<-data.frame(Site=c("YELL","WLOU","UKFS","SRER","SERC","RMNP","REDB","OAES","NOGP","MOAB","KONZ","HOPB","HEAL","DEJU","CUPE","ABBY","SJER","TEAK","NIWO","MLBS","BART","BLAN","BONA","CLBJ","DELA","DSNY","HARV","JERC","LENO","ONAQ","OSBS","SCBI","SOAP","TALL","UNDE","WREF"),
                    Type=c("D","D","D","C","D","D","D","D","D","D","D","D","C","D","D","C","D","C","C","D","D","D","D","D","D","C","D","D","D","C","C","D","D","D","D","C"))
sites<-str_match(lidar_tiles,"\\w+_(\\w+)_DP1")[,2]

parameter_df<-data.frame(Type=c("C","D"),max_cr_factor=c(0.2,0.9),exclusion=c(0.4,0.3))
site_df<-merge(site_df,parameter_df)

#Find LiDAR tiles
tile_df<-data.frame(Site=sites,Tile=lidar_tiles) %>% filter(!is.na(Site)) %>% filter(!str_detect(Tile,"Metadata")) %>% filter(str_detect(Tile,"ClassifiedPointCloud"))
batch_df<-merge(site_df,tile_df)

#Print number of tiles per site
print(paste("Total tiles:",nrow(batch_df)))

print("Number of tiles per site")
batch_df %>% group_by(Site) %>% summarize(n=n()) %>% as.data.frame() %>% arrange(desc(n)) 

#Random order
batch_df <- batch_df %>% sample_frac()
for(x in 1:length(batch_df$Tile)){
    print(x)
    tile = batch_df$Tile[x]
    sanitized_fn <- stringr::str_match(string = tile, pattern = "(\\w+).laz")[,2]
    lidar_year <- stringr::str_match(tile, "DP1.30003.001/neon-aop-products/(\\d+)/FullSite/")[,2]
    dst<-paste(save_dir,"/",lidar_year,"_",sanitized_fn,".csv",sep="")
    if(file.exists(dst)){
        next
    } else{
        skip_to_next <- FALSE
        tryCatch(detection_training_benchmark(path=tile,
               silva_cr_factor=batch_df$max_cr_factor[x],
               silva_exclusion=batch_df$exclusion[x],
               save_dir=save_dir,
               rgb_tiles=rgb_tiles),error = function(e) {skip_to_next <<- TRUE})
        if(skip_to_next) { next }
    }
}
