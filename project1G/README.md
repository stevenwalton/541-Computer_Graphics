# Compile 
---------
```
nvcc -o grayscaleShader{,.cu} `pkg-config --cflags --libs opencv`
```

# Issues
--------
To get pixel perfect images between the CPU and GPU we have to edit how the GPU
does its math. We have to match the fuse multiple add (FMA) functions. 

For example:

if we are trying to compute
```
a = a*a+b
```
The CPU will do
```
a = round(round(a*a)+b)
```
While CUDA will do
```
a = round(a*a+b)
```
We can fix this by using the `__fadd_rn` and `__fmul_rn` functions that cuda
provides. If we wanted to replicate the CPU function we would do
```
a = __fadd_rn(__fmul_rn(a,a),b);
```
For more information please refer to the 
[cuda documentation] (https://docs.nvidia.com/cuda/floating-point/index.html#axzz42SnDmIrm) 
relevant to the subject.
