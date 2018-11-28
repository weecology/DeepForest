#Profile Generator

from DeepForest import onthefly_generator,preprocess, config

import cProfile, pstats
cp = cProfile.Profile()

#Load data
DeepForest_config=config.load_config("train")
data=preprocess.load_data(DeepForest_config["training_csvs"],DeepForest_config["rgb_res"],DeepForest_config["lidar_path"])

#Split training and test data
train,test=preprocess.split_training(data,
                                     DeepForest_config,
                                     single_tile=DeepForest_config["single_tile"],
                                     experiment=None)

generator = onthefly_generator.OnTheFlyGenerator(
    data,
    train,
    batch_size=5,
    DeepForest_config=DeepForest_config,
    group_method="none",
shuffle_tile_epoch=True,
name="training")

cp.enable()

for i in range(generator.size()):
    raw_image    = generator.load_image(i)

cp.disable()

stats = pstats.Stats(cp)
stats.strip_dirs()
stats.sort_stats('cumulative', 'calls')
stats.print_stats(40)

#Dump
filename = 'profile.prof' 
stats.dump_stats(filename)