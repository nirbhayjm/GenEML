Datasets should be stored as single files in matlab export format ".mat" inside this directory. These are read using the the loadmat method in scipy.io.

It is assumed that on reading the .mat file the dictionary created will have the following keys:
 - 'Y_tr' -- training label matrix
 - 'X_tr' -- training feature matrix
 - 'Y_ts' -- testing label matrix
 - 'X_ts' -- testing feature matrix

The datasets used in this work are:
 - Bibtex
 - MediaMill
 - Eurlex-4K
 - MovieLens
 - RCV
 - Wiki10

 The datasets can be procured from [The Extreme Classification Repository: Multi-label Datasets & Code](http://manikvarma.org/downloads/XC/XMLRepository.html)