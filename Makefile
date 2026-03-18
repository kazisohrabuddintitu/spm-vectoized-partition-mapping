CXX       = g++
CXXSTD    = -std=c++17
INCLUDES  = -I src
WARNINGS  = -Wall -Wextra
DEPS      = src/keygen.hpp src/verify.hpp src/hpc_helpers.hpp

# ---- 32-bit hash (default) ----
BASELINE      = partition_map_baseline
AUTOVEC       = partition_map_autovec
AVX2          = partition_map_avx2

# ---- 64-bit hash (built with -DHASH64) ----
BASELINE_64   = partition_map_baseline_64
AUTOVEC_64    = partition_map_autovec_64
AVX2_64       = partition_map_avx2_64

.PHONY: all all64 clean vec-report vec-report-64

all: $(BASELINE) $(AUTOVEC) $(AVX2)

all64: $(BASELINE_64) $(AUTOVEC_64) $(AVX2_64)

# ===== 32-bit hash targets =====

$(BASELINE): src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -fno-tree-vectorize -o $@ $<

$(AUTOVEC): src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -mavx2 -fopt-info-vec-optimized -o $@ $<

$(AVX2): src/partition_map_avx2.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -mavx2 -o $@ $<

# ===== 64-bit hash targets =====

$(BASELINE_64): src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -DHASH64 -fno-tree-vectorize -o $@ $<

$(AUTOVEC_64): src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -DHASH64 -mavx2 -fopt-info-vec-optimized -o $@ $<

$(AVX2_64): src/partition_map_avx2.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -DHASH64 -mavx2 -o $@ $<

# ===== Vectorization reports =====

vec-report: src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -mavx2 \
	  -fopt-info-vec-optimized=vec_optimized_32.txt \
	  -fopt-info-vec-missed=vec_missed_32.txt \
	  -o $(AUTOVEC) $<
	@echo "32-bit hash reports: vec_optimized_32.txt, vec_missed_32.txt"

vec-report-64: src/partition_map.cpp $(DEPS)
	$(CXX) $(CXXSTD) $(INCLUDES) $(WARNINGS) -O3 -DHASH64 -mavx2 \
	  -fopt-info-vec-optimized=vec_optimized_64.txt \
	  -fopt-info-vec-missed=vec_missed_64.txt \
	  -o $(AUTOVEC_64) $<
	@echo "64-bit hash reports: vec_optimized_64.txt, vec_missed_64.txt"

clean:
	-rm -f $(BASELINE) $(AUTOVEC) $(AVX2) $(BASELINE_64) $(AUTOVEC_64) $(AVX2_64) *.txt
