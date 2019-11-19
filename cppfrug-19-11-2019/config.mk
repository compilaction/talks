# Project version
NAME    = prog

# CUDA arch
CUDA_GPU_ARCH ?= sm_52

# Compile flags
CXXFLAGS += -O3 -march=native
#CXXFLAGS += -Wall -Wextra -Werror -Wnull-dereference \
#            -Wdouble-promotion -Wshadow

# Language
CXXFLAGS += -std=c++17

# Includes
INCLUDES += -I$(CUDA_HOME)/include -I/opt/thrust
CXXFLAGS += $(INCLUDES)

# CUDA flags
CUFLAGS += --cuda-gpu-arch=$(CUDA_GPU_ARCH)

# Linker
LDFLAGS += -fPIC -O3
LDFLAGS += -L$(CUDA_HOME)/lib64
LDFLAGS += -lm -lcudart
