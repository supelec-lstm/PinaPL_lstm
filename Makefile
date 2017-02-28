OBJS = main.cpp cell.o functions.o network.o weights.o test.o
CC = clang++
CFLAGS = -std=c++11 -Ofast -c
LFLAGS = -std=c++11 -Ofast

Build : $(OBJS)
	$(CC) $(LFLAGS) -o Build $(OBJS)

cell.o : cell.hpp cell.cpp weights.hpp functions.hpp
	$(CC) $(CFLAGS) cell.cpp -o cell.o

network.o : network.hpp network.cpp
	$(CC) $(CFLAGS) network.cpp -o network.o

weights.o : weights.cpp weights.hpp
	$(CC) $(CFLAGS) weights.cpp -o weights.o

functions.o : functions.cpp functions.hpp
	$(CC) $(CFLAGS) functions.cpp -o functions.o

test.o : test.cpp test.hpp
	$(CC) $(CFLAGS) test.cpp -o test.o


clean:
	\rm *.o *~ build

