The code implements EM fitting of a mixture of gaussians on the two-dimensional data set

K: the numbers of mixtures,

threshold: when differences between two loglikelihood of train set less than threshold, it will stop learning

tiedmode: when it is set as True, it uses tied covariance matrices, on the other hand, it uses separate covariance matrices

Put code and dataset in same directory and run the code, 
it will output as following form with two loglikelihood-iter figures or train and validation set:
train iter= 0, loglikelihood for train set is -3290.13179149
loglikelihood for validation set is -398.631366221
train iter= 1, loglikelihood for train set is -3195.14630898
loglikelihood for validation set is -362.488108283
train iter= 2, loglikelihood for train set is -2986.14980884
loglikelihood for validation set is -353.156913733
train iter= 3, loglikelihood for train set is -2936.71947849
loglikelihood for validation set is -349.200342199
train iter= 4, loglikelihood for train set is -2917.52512391
loglikelihood for validation set is -346.861230337
train iter= 5, loglikelihood for train set is -2906.92805894
loglikelihood for validation set is -344.983317581
train iter= 6, loglikelihood for train set is -2898.63541773
loglikelihood for validation set is -343.0875401
train iter= 7, loglikelihood for train set is -2890.07518067
loglikelihood for validation set is -340.895169091
train iter= 8, loglikelihood for train set is -2879.71800769
loglikelihood for validation set is -338.197863719
train iter= 9, loglikelihood for train set is -2866.26881289
loglikelihood for validation set is -334.847758931
train iter= 10, loglikelihood for train set is -2848.45795334
loglikelihood for validation set is -330.833680603
train iter= 11, loglikelihood for train set is -2825.30919936
loglikelihood for validation set is -326.444999176
train iter= 12, loglikelihood for train set is -2797.19027991
loglikelihood for validation set is -322.437065267
train iter= 13, loglikelihood for train set is -2767.89569087
loglikelihood for validation set is -319.802391523
train iter= 14, loglikelihood for train set is -2745.34213401
loglikelihood for validation set is -318.757216632
train iter= 15, loglikelihood for train set is -2734.24413847
loglikelihood for validation set is -318.527767995
train iter= 16, loglikelihood for train set is -2730.48889841
loglikelihood for validation set is -318.511427666
train iter= 17, loglikelihood for train set is -2729.15872112
loglikelihood for validation set is -318.540151387
train iter= 18, loglikelihood for train set is -2728.4466289
loglikelihood for validation set is -318.587672855
train iter= 19, loglikelihood for train set is -2727.94635872
loglikelihood for validation set is -318.647791108
train iter= 20, loglikelihood for train set is -2727.56599487
loglikelihood for validation set is -318.715933365
train iter= 21, loglikelihood for train set is -2727.27072612
loglikelihood for validation set is -318.788435604
train iter= 22, loglikelihood for train set is -2727.03902276
loglikelihood for validation set is -318.862697822
train iter= 23, loglikelihood for train set is -2726.85521276
loglikelihood for validation set is -318.937015314
train iter= 24, loglikelihood for train set is -2726.70752825
loglikelihood for validation set is -319.010338027
train iter= 25, loglikelihood for train set is -2726.58709909
loglikelihood for validation set is -319.082070359
train iter= 26, loglikelihood for train set is -2726.48724705
K = 6 ,final centroids are:
[[ 1.13195667  1.68524072]
 [-1.01116345  0.72369988]
 [-0.97458202 -0.98293205]
 [ 1.53046277 -0.98851363]
 [-1.028488    1.26780142]
 [ 0.92922737  1.3410762 ]]

