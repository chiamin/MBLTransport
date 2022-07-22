# MBLTransport
Compute transport properties by quench dynamics using MPS.

## Compile
1. Clone another repository for the utility functions\
``
 git clone https://github.com/chiamin/itensor.utility
``

2. Specify the variable `MYDIR` in **Makefile** to the path of the above utility repository

3. Use the command `make` to compile the code\
There are two source code files, **mu_quench.cc** and **den_quench.cc**. Choose which one to compile in **Makefile**.

## Run the executive file
Run the program by the command
```
./mu_quench.exe input
```
where **input** is an input file. An example of input file is given as **input**.

## Analysis data
One can use the command
```
./mu_quench.exe input > output
```
to save the output messages into a file **output**. Then use
```
python analysis.py output
```
to plot the figures.
