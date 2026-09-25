# Compiler and flags
CXX      := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra -O3

# Target binary name and source files
TARGET   := model_test
SRCS     := model_test.cpp llama_model.cpp model_loader.cpp dequant.cpp ops.cpp infer_state.cpp forward.cpp

# Default rule: built when you just type `make`
all: $(TARGET)

# Rule to compile the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS)

# Cleanup build artifacts
clean:
	rm -f $(TARGET)

# Declare targets that do not represent actual files on disk
.PHONY: all clean
