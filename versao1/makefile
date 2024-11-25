CXX = g++

CXXFLAGS = -Wall -g

SOURCES = grafo.cpp sort.cpp

OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = programa

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
