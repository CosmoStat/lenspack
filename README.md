# lenspack
[![Build Status](https://travis-ci.org/austinpeel/lenspack.svg?branch=master)](https://travis-ci.org/austinpeel/lenspack) [![Coverage Status](https://coveralls.io/repos/github/austinpeel/lenspack/badge.svg?branch=master)](https://coveralls.io/github/austinpeel/lenspack?branch=master)

---
> Author: <a href="http://www.cosmostat.org/people/austin-peel" target="_blank" style="text-decoration:none; color: #F08080">Austin Peel</a>  
> Email: <a href="mailto:austin.peel@cea.fr" style="text-decoration:none; color: #F08080">austin.peel@cea.fr</a>  
> Year: 2019  
---

This repository is a collection of python codes useful for the weak-lensing
analysis of galaxy catalogs and shear/convergence maps. The full documentation
can be found [here](https://austinpeel.github.io/lenspack/index.html "lenspack documentation").

## Contents

In progress.

## Dependencies

* numpy
* scipy
* astropy
* emcee (optional)

## Examples

### Peak detection

Suppose you have a galaxy catalog `cat` containing sky position columns `ra` and `dec`, along with ellipticity components `e1` and `e2`. You can bin the galaxies into pixels, invert the shear to obtain convergence (Kaiser & Squires, 1993), detect peaks above a given threshold, and plot the result as follows.

```python
import matplotlib.pyplot as plt
from lenspack.utils import bin2d
from lenspack.image.inversion import ks93
from lenspack.peaks import find_peaks2d

# Bin ellipticity values according to galaxy position into a 256 x 256 map
e1map, e2map = bin2d(cat['ra'], cat['dec'], v=(cat['e1'], cat['e2']), npix=256)

# Kaiser-Squires inversion
kappaE, kappaB = ks93(e1map, e2map)

# Detect peaks on the convergence E-mode map
x, y, h = find_peaks2d(kappaE, threshold=0.05, include_border=True)

# Plot peak positions over the convergence
fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
plot2d(kappaE, cmap='bone', fig=fig, ax=ax)
ax.scatter(y, x, s=5, c='orange', alpha=0.7) # reverse x and y due to array indexing
ax.set_axis_off()
plt.show()
```

<p align="left">
<img src="https://github.com/austinpeel/lenspack/blob/master/examples/figures/peaks.png" alt="peaks" width="400"/>
</p>


### Wavelet transform
