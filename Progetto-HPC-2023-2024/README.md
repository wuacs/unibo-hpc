# Instructions

## How to compile
In the first directory there is a Makefile, which can be used in the following manners:

- `make` (without target) will create two executables _omp-circles_, which parallelizes using OpenMP and _cuda-circles_ which is based on CUDA.
- `make cuda` will create just the executable based on CUDA.
- `make omp` will create just the executable based on OpenMP.
- `make movie` will create two executables _omp-circles-movie_ and _cuda-circles-movie_ which, when executed, additionally create an files containing a visual rapresentation of the iterations of the program. Refer to [On Movie Executables](#on-movie-executables) for more info.
- `make movie_cuda` will create just cuda-circles-movie and the _movie_ folder. (refer to _make movie_ for info)
- `make movie_omp` will create just omp-circles-movie (refer to _make movie_ for info).
- `make clean` will kill __all__ files or directories created by any of the commands before(or created by you manually but with same name).
## How to execute

After consulting the [How to compile](#how-to-compile) section and launching appropriate `make` commands you will now have some executables in the same folder this README.md is.

To execute any executable just use:

        ./[pathToExecutable]/executablename [ncircles [iterations]] 

where:

- `ncircles` is the number of pseudorandom circles to spawn. It must be a positive C integer. Default value is 10.000.
- `iterations`is the number of iterations the program will perform for each circle. It must be a positive C integer. It's default value is 20.

> **WARNING**: launching any executable with large enough number of circles and/or iterations, especially if launching a *-movie, could take more than a minute. Refer to Relazione.pdf for more information.

## On Movie executables

A lot of `circles-xxxxx.gp` files will be produced; these files must
be processed using `gnuplot` to create individual frames:

        for f in *.gp; do gnuplot "$f"; done

and then assembled to produce the movie `circles.avi`:

        ffmpeg -y -i "circles-%05d.png" -vcodec mpeg4 circles.avi