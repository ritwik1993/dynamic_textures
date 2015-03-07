PROG=dynamic_textures
CC=g++
CPPFLAGS=-c -Wall
LDFLAGS=`pkg-config --cflags opencv` `pkg-config --libs opencv`
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)

$(PROG): $(OBJECTS)
	$(CC) -o $(PROG) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CPPFLAGS) -o $@ $<

clean:
	rm -rf *.o final *~

rebuild: clean $(PROG)
