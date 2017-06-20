import libatrous
import numpy as np
import time

a = np.zeros((16000,16000),np.float32)
print "zeros will do...",a.shape

#setting grid
if 1:
    print "2 dimensions..."
    libatrous.set_grid(1,1)
    print "3 dimensions..."
    libatrous.set_grid(1,1,1)

#kernels
if 1:
    print "Getting SPL5 kernel array..."
    kernel = libatrous.choose_kernel(libatrous.SPL5)
    print kernel

n_test = 10
if 1:
    print "testing scale"
    kernel_index = 1
    scale = 8
    t = time.time()
    b = libatrous.scale(a,kernel,scale)
    print "took",time.time()-t
    print a.shape
    print b.shape
    print a.ndim
    print b.ndim
    print b.shape == a.shape

if 1:
    print "testing stack"
    kernel_index = 0
    n_scales = 12
    t = time.time()
    scales = libatrous.stack(a,kernel,n_scales)
    print scales.shape
    print "took",time.time()-t
    print scales.shape[1::] == a.shape
    print scales.shape[0] == n_scales + 1
    print np.sum(scales) == 0
    del scales
    print
    time.sleep(2)

if 1:
    print "testing iterscale"
    kernel_index = 0
    n_scales = 12
    t = time.time()

    c = a.copy()
    scales = [None]*(n_scales+1)
    for i in range(n_scales):
        b,c = libatrous.iterscale(c,kernel,i)
        scales[i] = b
    scales[n_scales] = c
    print "took",time.time()-t

