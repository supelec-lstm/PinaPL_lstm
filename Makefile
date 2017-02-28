OBJS = cell.o functions.o network.o weights.o test.o
CC = gcc
DEBUG = -g
CFLAGS = -Wall -c -std=c++11 $(DEBUG)
LFLAGS = -Wall -std=c++11 $(DEBUG)

build : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o build

cell.o : cell.hpp cell.cpp weights.hpp functions.hpp
	$(CC) $(CFLAGS) cell.cpp

network.o : network.hpp network.cpp
	$(CC) $(CFLAGS) network.cpp

weights.o : weights.cpp weights.hpp
	$(CC) $(CFLAGS) weights.cpp

functions.o : functions.cpp functions.hpp
	$(CC) $(CFLAGS) functions.cpp

test.o : test.cpp test.hpp
	$(CC) $(CFLAGS) test.cpp


clean:
	\rm *.o *~ build

