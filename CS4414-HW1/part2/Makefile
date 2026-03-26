CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall
TARGET = main
SOURCES = main.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: clean
