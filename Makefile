.POSIX:
.SUFFIXES: .cpp .hpp .h .cu .o .d .asm

include config.mk

# Sources
CXXSRC = $(shell find src -name "*.cpp")
CUSRC  = $(shell find src -name "*.cu")

# Objects
CXXOBJ = $(CXXSRC:.cpp=.o)
CUOBJ  = $(CUSRC:.cu=.o)

OBJ    = $(CXXOBJ) $(CUOBJ)

# Dependency files
DEPS   = $(OBJ:.o=.d)

all: $(NAME)

# Compilation
.cpp.o:
	$(CXX) $(CXXFLAGS) -MMD -c -o $@ $<
.cu.o:
	$(CU) $(CXXFLAGS) $(CUFLAGS) -MMD -c -o $@ $<

# Linking
$(NAME): $(OBJ)
	$(CU) -o $@ $(OBJ) $(LDFLAGS)

# Dependencies
-include $(DEPS)

debug: CXXFLAGS += -DDEBUG -g
debug: $(NAME)

clean:
	rm -f $(NAME) $(NAME).asm $(OBJ) $(DEPS)

run: $(NAME)
	./$(NAME)

remote: $(NAME)
	scp $(NAME) $(REMOTE_EXEC_HOST):/tmp/$(NAME)
		ssh $(REMOTE_EXEC_HOST) /tmp/$(NAME)

dump: $(NAME)
	objdump -dC $(NAME) > $(NAME).asm

.PHONY: all clean slurm run dump
