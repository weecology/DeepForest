"""Holds metrics"""

import numpy as np
from scipy import stats

# x y z i r
# 0 1 2 3 4

class CloudData:
    """Maybe just use cloud info?"""
    def __init__(self, path):
        import laspy
        self.path = path
        self.cloud = laspy.file.File(self.path)
        self.array = np.column_stack((self.cloud.x, self.cloud.y, self.cloud.z, self.cloud.intensity, self.cloud.return_num))

    def minimum(self, dim):
        """Where dimension is 0 1 2 3 4"""
        return self.array[:,dim].min()

    def max(self, dim):
        return self.array[:,dim].max()

    def mode(self, dim, list_out=False):
        mode_list = stats.mode(self.array[:,dim])[0]
        if len(mode_list) > 1:
            print("Warning: only first mode returned of a list of modes. Use list_out=True when calling mode function\
                  to return a list of modes.")
            if list_out == False:
                return mode_list[0]
            else:
                return mode_list
        else:
            return mode_list[0]

    def mean(self, dim):
        """Arithmetic mean."""
        return np.mean(self.array[:,dim])

    def std(self, dim):
        """Population standard deviation."""
        return np.std(self.array[:,dim])

    def var(self, dim):
        """Population variance."""
        return np.var(self.array[:, dim])

    def cvar(self, dim):
        """Coefficient of variation."""
        return self.std(dim) / float(self.mean(dim))

    def interquartile(self, dim):
        return stats.iqr(self.array[:, dim])

    def skewness(self, dim):
        return stats.skew(self.array[:, dim])

    def kurtosis(self, dim):
        return stats.kurtosis(self.array[:, dim])

    def AAD(self, dim):
        """Average (mean) absolute deviation."""
        return np.mean(np.absolute(self.array[:, dim] - np.mean(self.array[:, dim])))

    def MAD_median(self, dim):
        """Median absolute deviation."""
        return np.median(np.absolute(self.array[:, dim] - np.median(self.array[:, dim])))

    def MAD_mode(self, dim):
        #FIXME
        #TODO: Add handling for lists of modes.
        return self.mode(np.absolute(self.array[:, dim] - self.mode(self.array[:, dim])))

    def l_moment(self, l, dim):
        #TODO: this
        pass

    def l_skew(self, dim):
        #TODO: this
        pass

    def l_kurtosis(self, dim):
        #TODO: this
        pass

    def percentile(self, dim, pct):
        return np.percentile(self.array[:, dim], pct)

    def canopy_relief_ratio(self):
        #TODO: check if this is correct
        return (self.mean(2) - self.min(2)) / float((self.max(2)) - self.min(2))

    def quadratic_mean(self, dim):
        #TODO: this
        # https://en.wikipedia.org/wiki/Root_mean_square
        pass

    def cubic_mean(self, dim):
        #TODO: this
        # https://en.wikipedia.org/wiki/Cubic_mean
        pass

    ## Non-Dimensional

    def total_count(self, dim=0):
        """Total number of records, default to first column."""
        return self.array[:,dim].shape[0]

    def return_count(self, n = 0):
        """Returns total number of nth returns."""
        if n == 'all':
            return self.total_count()
        else:
            unique, counts = np.unique(self.array[:,4], return_counts=True)
            return dict(zip(unique, counts)).get(n)

    def above(self, threshold, n_return = 0):
        """Returns number of returns above a given z-threshold (in same units as vertical datum)."""
        if n_return == 0:
            all_z = self.array[:,2]
            filter = (all_z > threshold).sum()
            return filter
        else:
            #TODO: this
            pass

    ## Crown
    def pct_above(self, threshold, n_return = 0, d_return = 0):
        return self.above(threshold, n_return) / float(self.return_count(d_return))


    def pct_abv_mean(self, n_return = 0):
        return (self.above(self.mean(2), n_return) / float(self.total_count()))


    def pct_abv_mode(self):
        # FIXME
        #return (self.above(self.mode(2), n_return) / float(self.total_count()))
        pass

    def returns_abv_mode(self):
        # FIXME
        return self.above(self.mode(2))

    def returns_abv_mean(self):
        return self.above(self.mean(2))

    def returns_abv_mean_totf(self):
        return self.returns_abv_mean() / float(self.return_count(1))

    def returns_abv_mode_tot(self):
        return self.returns_abv_mode() / float(self.return_count(1))

    def first_abv_mean(self):
        return self.above(self.mean(2), n_return=1)

    def first_abv_mode(self):
        return self.above(self.mode(2), n_return=1)

    def total_returns(self):
        pass