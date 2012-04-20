All : main.o IC.o CL.o FD.o
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  OculOs main.o IC.o CL.o FD.o
main.o :main.cpp IC.h CL.h FD.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  main.o -c main.cpp
IC.o: IC.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  IC.o -c IC.cpp
CL.o: CL.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  CL.o -c CL.cpp
FD.o: FD.cpp
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  FD.o -c FD.cpp
