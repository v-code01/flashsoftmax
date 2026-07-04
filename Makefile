CXX      ?= clang++
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -Werror -Isrc
BIN := bin
SRC := src
.PHONY: all test clean
all: $(BIN)/bench
$(BIN)/bench: $(SRC)/bench.cpp $(SRC)/softmax.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(SRC)/bench.cpp -o $@
$(BIN)/test_softmax: tests/test_softmax.cpp $(SRC)/softmax.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) tests/test_softmax.cpp -o $@
test: $(BIN)/test_softmax
	$(BIN)/test_softmax
$(BIN):
	mkdir -p $(BIN)
clean:
	rm -rf $(BIN)
