CAUTION!

mpi-mandelbrot.c is NOT parallelized, but is used for testing the correctness of the parallelized versions.

mpi-mandelbrot-gather has 2 limitations: 

1) The first, which is controlled at command line, does not let you input a height which is not 
properly divisible by the number of processes used.
2) The second, because everything is done in integers, specifically image's bytes trasmission and mapping, if you input a large
number(anything above 100k) as the height, it will crash. X-axis length is hard-coded has x-axis = y-axis * 1.4 which means for y-axis=100K it 
yields x-axis=140K but since an image is transitted as bytes we index, with an integer, x-axis * y-axis * 3(RGB values) which yields 42.000.000.000.

mpi-mandelbrot-gatherv.c has as only known limitation the second limitation of mpi-mandelbrot-gather.c.