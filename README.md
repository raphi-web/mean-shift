# mean-shift-numba

A implementation of the mean-shift algorithm for images, accelerated with Numba.  
Thanks to Numba with parallel processing the clustering is fast.  


Maybe I will make this a fully developed librar later but for now treat this just as an example code.


Pramarely written for multi-band satellite images
`mean_shift.py` is for clustering based on RGB-Color images.
`mean_shift_spatial.py` if for clustering images with an arbetrary number of bands, it also is a windowed approach that allows to pass distance weights.
