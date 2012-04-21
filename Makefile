All : main.o IC.o CL.o FD.o FL.o
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  OculOs main.o IC.o CL.o FD.o FL.o
main.o :main.cpp IC.h CL.h FD.h FL.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  main.o -c main.cpp
IC.o: IC.cpp IC.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  IC.o -c IC.cpp
CL.o: CL.cpp CL.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  CL.o -c CL.cpp
FD.o: FD.cpp FD.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  FD.o -c FD.cpp
FL.o: FL.cpp FL.h
	g++ -g -pg -Wall `pkg-config --libs --cflags opencv` -o  FL.o -c FL.cpp
