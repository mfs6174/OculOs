All : main.o IC.o
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  OculOs main.o IC.o
main.o :main.cpp IC.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  main.o -c main.cpp
IC.o: IC.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  IC.o -c IC.cpp
