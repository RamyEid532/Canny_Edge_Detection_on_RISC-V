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

# Added -flax-vector-conversions to fix type errors
RV_CXXFLAGS = -Wall -Wextra -std=c++17 -march=rv64gcv -mabi=lp64d -O0 -g -I$(INC_DIR) \
              --sysroot=/home/ramy_eid/riscv-toolchain/riscv64-unknown-elf \
              -flax-vector-conversions

VLEN ?= 128
QEMU_FLAGS = -cpu rv64,v=true,vlen=$(VLEN)

# ==========================================
# Source Files
# ==========================================
ALL_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CORE_SRCS = $(filter-out $(SRC_DIR)/main.cpp, $(ALL_SRCS))
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

$(HOST_BUILD)/host_tests: $(TEST_MAIN) $(CORE_SRCS)
	@mkdir -p $(HOST_BUILD)
	$(HOST_CXX) $(HOST_CXXFLAGS) $^ $(HOST_LDFLAGS) -o $@

canny_rv: $(RV_BUILD)/canny_rv
	@echo "RISC-V binary built at $(RV_BUILD)/canny_rv"

$(RV_BUILD)/canny_rv: $(RV_MAIN) $(CORE_SRCS)
	@mkdir -p $(RV_BUILD)
	$(RV_CXX) $(RV_CXXFLAGS) $^ -o $@

run: canny_rv
	@echo "Executing on QEMU with VLEN=$(VLEN)..."
	$(QEMU) $(QEMU_FLAGS) ./$(RV_BUILD)/canny_rv

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
