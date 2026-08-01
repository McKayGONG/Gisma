# ********     Author: Lijun Chang    ******
# ******** Email: ljchang@outlook.com ******
#
CC=g++ -O3
CFLAGS=-c -I. -std=c++11 -pthread

all: ged

ged: .obj/main.o .obj/Application.o
	${CC} .obj/main.o .obj/Application.o -o ged -pthread
	rm -rf *o .obj/
	mkdir -p .obj

.obj/main.o: main.cpp | .obj
	${CC} ${CFLAGS} -o .obj/main.o main.cpp

.obj/Application.o: Application.cpp | .obj
	${CC} ${CFLAGS} -o .obj/Application.o Application.cpp

clean:
	rm -rf *o .obj/
	mkdir -p .obj

.obj:
	mkdir -p .obj
