All : main.o IC.o CL.o
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  OculOs main.o IC.o CL.o
main.o :main.cpp IC.h CL.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  main.o -c main.cpp
IC.o: IC.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  IC.o -c IC.cpp
CL.o: CL.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  CL.o -c CL.cpp
