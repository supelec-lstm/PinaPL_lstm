#!/bin/sh
# clang++ compilation script

# object files generation
clang++ -std=c++11 -Ofast -c cell.cpp -o cell.o
clang++ -std=c++11 -Ofast -c functions.cpp -o functions.o
clang++ -std=c++11 -Ofast -c network.cpp -o network.o
clang++ -std=c++11 -Ofast -c test.cpp -o test.o
clang++ -std=c++11 -Ofast -c weights.cpp -o weights.o

# linking and building
# -lz option is necessary to interact with .gz files (IDX parser)
# -Weverything option shows all warnings
# -std=c++11 option forces c++11 compatibility

clang++ -std=c++11 -Ofast -o Build main.cpp cell.o functions.o network.o test.o weights.o 

# cleaning object files
rm *.o
