CXX = g++
NVCC = nvcc

CXXFLAGS = -Wall -g
NVCCFLAGS = -arch=sm_30

SOURCES = grafo.cu sort.cpp

OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE = programa

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
