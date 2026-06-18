# ==========================================
# Compilers and Tools
# ==========================================
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++
QEMU = qemu-riscv64

# ==========================================
# Directories
# ==========================================
INC_DIR = include
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
HOST_BUILD = $(BUILD_DIR)/host
RV_BUILD = $(BUILD_DIR)/rv

# ==========================================
# Flags
# ==========================================
HOST_CXXFLAGS = -Wall -Wextra -std=c++17 -O0 -g -I$(INC_DIR)
HOST_LDFLAGS = -lgtest -lgtest_main -pthread

RV_CXXFLAGS = -Wall -Wextra -std=c++17 -march=rv64gcv -mabi=lp64d $(OPT) -g -I$(INC_DIR) \
	      --sysroot=/home/khaled-abdeluziz/riscv-toolchain/riscv64-unknown-elf

VLEN ?= 128
QEMU_FLAGS = -cpu rv64,v=true,vlen=$(VLEN)

# ==========================================
# Source Files
# ==========================================
# 1. Grab EVERY .cpp file in the src directory
ALL_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# 2. Filter out main.cpp so we don't get "multiple definition of main" in tests
CORE_SRCS = $(filter-out $(SRC_DIR)/main.cpp, $(ALL_SRCS))

# 3. syscalls.cpp implements newlib's bare-metal stubs (_write, _read, _open,
#    _sbrk, ...). The HOST build uses glibc, which already defines all of
#    these — linking syscalls.cpp into the host tests causes
#    "multiple definition of `_write'" (and friends) errors. So we build a
#    SEPARATE source list for the host that excludes it.
HOST_CORE_SRCS = $(filter-out $(SRC_DIR)/syscalls.cpp, $(CORE_SRCS))

RV_MAIN = $(SRC_DIR)/main.cpp
TEST_MAIN = $(TEST_DIR)/unit_tests.cpp

# ==========================================
# Targets
# ==========================================
.PHONY: all test canny_rv run clean

all: test canny_rv

test: $(HOST_BUILD)/host_tests
	@echo "Running host tests natively..."
	./$(HOST_BUILD)/host_tests

$(HOST_BUILD)/host_tests: $(TEST_MAIN) $(HOST_CORE_SRCS)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_CXXFLAGS) $^ $(HOST_LDFLAGS) -o $@

canny_rv: $(RV_BUILD)/canny_rv
	@echo "RISC-V binary built at $(RV_BUILD)/canny_rv"

# CORE_SRCS here DOES include syscalls.cpp — the RISC-V binary needs it
# to bridge newlib calls to QEMU's Linux syscalls.
$(RV_BUILD)/canny_rv: $(RV_MAIN) $(CORE_SRCS)
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_CXXFLAGS) $^ -o $@

run: canny_rv
	@echo "Executing on QEMU with VLEN=$(VLEN)..."
	$(QEMU) $(QEMU_FLAGS) $(RV_BUILD)/canny_rv $(abspath $(IMG))

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
