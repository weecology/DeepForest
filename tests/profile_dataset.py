# Profile the dataset class
import cProfile
import os
import pstats

from deepforest import dataset
from deepforest import get_data


def run():
    csv_file = get_data("OSBS_029.csv")
    root_dir = os.path.dirname(csv_file)

    ds = dataset.TreeDataset(csv_file=csv_file,
                             root_dir=root_dir,
                             transforms=dataset.get_transform(augment=True))

    for x in range(1000):
        next(iter(ds))


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    stats.dump_stats('dataset.prof')
