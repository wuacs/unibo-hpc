Since CUDA 12 cuda-memcheck is deprecated.
Now use `compute-sanitizer --tool memcheck app_name`.

The program bugged.cu has a memory leak which can be found using the command above.