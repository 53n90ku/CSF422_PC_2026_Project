CXX      := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra
TARGET   := abc
SRCS     := main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) models/gaussian_model.h models/sir_model.h engine/abc_engine.h rng.h
	$(CXX) $(CXXFLAGS) -o $@ $(SRCS)

clean:
	rm -f $(TARGET)
