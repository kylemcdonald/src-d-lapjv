#pragma once

#include <cassert>
#include <cstdio>
#include <limits>
#include <memory>

#ifdef __aarch64__
  // For ARM64, use ARM-specific NEON headers
  #include <arm_neon.h>
#else
  // x86 specific headers
  #include <immintrin.h>
#endif

#ifdef __GNUC__
#define always_inline __attribute__((always_inline)) inline
#define restrict __restrict__
#elif _WIN32
#define always_inline __forceinline
#define restrict __restrict
#else
#define always_inline inline
#define restrict
#endif


template <typename idx, typename cost>
always_inline std::tuple<cost, cost, idx, idx>
find_umins_regular(
    idx dim, idx i, const cost *restrict assign_cost,
    const cost *restrict v) {
  const cost *local_cost = &assign_cost[i * dim];
  cost umin = local_cost[0] - v[0];
  idx j1 = 0;
  idx j2 = -1;
  cost usubmin = std::numeric_limits<cost>::max();
  for (idx j = 1; j < dim; j++) {
    cost h = local_cost[j] - v[j];
    if (h < usubmin) {
      if (h >= umin) {
        usubmin = h;
        j2 = j;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = j;
      }
    }
  }
  return std::make_tuple(umin, usubmin, j1, j2);
}

// These are not constexpr because of typename idx
#define FLOAT_MIN_DIM 64
#ifdef __aarch64__
  #define DOUBLE_MIN_DIM 32  // NEON double operations are efficient on M1
#else
  #define DOUBLE_MIN_DIM 100000  // 64-bit code is actually always slower
#endif

// Add NEON optimized version
template <typename idx>
always_inline std::tuple<float, float, idx, idx>
find_umins_neon(
    idx dim, idx i, const float *restrict assign_cost,
    const float *restrict v) {
  if (dim < FLOAT_MIN_DIM) {
    return find_umins_regular(dim, i, assign_cost, v);
  }
  const float *local_cost = assign_cost + i * dim;
  
  // Use vld1q_f32_x4 for better vectorization when possible
  float32x4_t uminvec = vdupq_n_f32(std::numeric_limits<float>::max());
  float32x4_t usubminvec = vdupq_n_f32(std::numeric_limits<float>::max());
  int32x4_t j1vec = vdupq_n_s32(-1);
  int32x4_t j2vec = vdupq_n_s32(-1);
  int32x4_t idxvec = {0, 1, 2, 3};
  
  // Process 16 elements at once when possible
  for (idx j = 0; j < dim - 15; j += 16) {
    float32x4x4_t acvec = vld1q_f32_x4(local_cost + j);
    float32x4x4_t vvec = vld1q_f32_x4(v + j);
    
    // Process each quarter
    for (int q = 0; q < 4; q++) {
      float32x4_t h = vsubq_f32(acvec.val[q], vvec.val[q]);
      
      uint32x4_t cmp = vcleq_f32(h, uminvec);
      usubminvec = vbslq_f32(cmp, uminvec, usubminvec);
      j2vec = vbslq_s32(cmp, j1vec, j2vec);
      uminvec = vbslq_f32(cmp, h, uminvec);
      j1vec = vbslq_s32(cmp, idxvec, j1vec);
      
      cmp = vandq_u32(vmvnq_u32(cmp), vcltq_f32(h, usubminvec));
      usubminvec = vbslq_f32(cmp, h, usubminvec);
      j2vec = vbslq_s32(cmp, idxvec, j2vec);
      
      idxvec = vaddq_s32(idxvec, vdupq_n_s32(4));
    }
  }

  // Process remaining vectors of 4
  for (idx j = dim & ~15; j < dim - 3; j += 4) {
    float32x4_t acvec = vld1q_f32(local_cost + j);
    float32x4_t vvec = vld1q_f32(v + j);
    float32x4_t h = vsubq_f32(acvec, vvec);
    
    uint32x4_t cmp = vcleq_f32(h, uminvec);
    usubminvec = vbslq_f32(cmp, uminvec, usubminvec);
    j2vec = vbslq_s32(cmp, j1vec, j2vec);
    uminvec = vbslq_f32(cmp, h, uminvec);
    j1vec = vbslq_s32(cmp, idxvec, j1vec);
    
    cmp = vandq_u32(vmvnq_u32(cmp), vcltq_f32(h, usubminvec));
    usubminvec = vbslq_f32(cmp, h, usubminvec);
    j2vec = vbslq_s32(cmp, idxvec, j2vec);
    
    idxvec = vaddq_s32(idxvec, vdupq_n_s32(4));
  }

  // Ensure proper alignment for storing results
  alignas(16) float uminmem[4], usubminmem[4];
  alignas(16) int32_t j1mem[4], j2mem[4];
  
  vst1q_f32(uminmem, uminvec);
  vst1q_f32(usubminmem, usubminvec);
  vst1q_s32(j1mem, j1vec);
  vst1q_s32(j2mem, j2vec);

  idx j1 = -1, j2 = -1;
  float umin = std::numeric_limits<float>::max(),
        usubmin = std::numeric_limits<float>::max();
  for (int vi = 0; vi < 4; vi++) {
    float h = uminmem[vi];
    if (h < usubmin) {
      idx jnew = j1mem[vi];
      if (h >= umin) {
        usubmin = h;
        j2 = jnew;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = jnew;
      }
    }
  }
  for (int vi = 0; vi < 4; vi++) {
    float h = usubminmem[vi];
    if (h < usubmin) {
      usubmin = h;
      j2 = j2mem[vi];
    }
  }
  for (idx j = dim & 0xFFFFFFF8u; j < dim; j++) {
    float h = local_cost[j] - v[j];
    if (h < usubmin) {
      if (h >= umin) {
        usubmin = h;
        j2 = j;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = j;
      }
    }
  }
  return std::make_tuple(umin, usubmin, j1, j2);
}

template <typename idx>
always_inline std::tuple<double, double, idx, idx>
find_umins_neon_double(
    idx dim, idx i, const double *restrict assign_cost,
    const double *restrict v) {
  if (dim < DOUBLE_MIN_DIM) {
    return find_umins_regular(dim, i, assign_cost, v);
  }
  const double *local_cost = assign_cost + i * dim;
  
  float64x2_t uminvec = vdupq_n_f64(std::numeric_limits<double>::max());
  float64x2_t usubminvec = vdupq_n_f64(std::numeric_limits<double>::max());
  int64x2_t j1vec = vdupq_n_s64(-1);
  int64x2_t j2vec = vdupq_n_s64(-1);
  int64x2_t idxvec = {0, 1};
  
  for (idx j = 0; j < dim - 1; j += 2) {
    float64x2_t acvec = vld1q_f64(local_cost + j);
    float64x2_t vvec = vld1q_f64(v + j);
    float64x2_t h = vsubq_f64(acvec, vvec);
    
    uint64x2_t cmp = vcleq_f64(h, uminvec);
    usubminvec = vbslq_f64(cmp, uminvec, usubminvec);
    j2vec = vbslq_s64(cmp, j1vec, j2vec);
    uminvec = vbslq_f64(cmp, h, uminvec);
    j1vec = vbslq_s64(cmp, idxvec, j1vec);
    
    cmp = vandq_u64(vmvnq_u64(cmp), vcltq_f64(h, usubminvec));
    usubminvec = vbslq_f64(cmp, h, usubminvec);
    j2vec = vbslq_s64(cmp, idxvec, j2vec);
    
    idxvec = vaddq_s64(idxvec, vdupq_n_s64(2));
  }

  alignas(16) double uminmem[2], usubminmem[2];
  alignas(16) int64_t j1mem[2], j2mem[2];
  
  vst1q_f64(uminmem, uminvec);
  vst1q_f64(usubminmem, usubminvec);
  vst1q_s64(j1mem, j1vec);
  vst1q_s64(j2mem, j2vec);

  idx j1 = -1, j2 = -1;
  double umin = std::numeric_limits<double>::max(),
         usubmin = std::numeric_limits<double>::max();
  for (int vi = 0; vi < 2; vi++) {
    double h = uminmem[vi];
    if (h < usubmin) {
      idx jnew = j1mem[vi];
      if (h >= umin) {
        usubmin = h;
        j2 = jnew;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = jnew;
      }
    }
  }
  for (int vi = 0; vi < 2; vi++) {
    double h = usubminmem[vi];
    if (h < usubmin) {
      usubmin = h;
      j2 = j2mem[vi];
    }
  }
  for (idx j = dim & 0xFFFFFFFEu; j < dim; j++) {
    double h = local_cost[j] - v[j];
    if (h < usubmin) {
      if (h >= umin) {
        usubmin = h;
        j2 = j;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = j;
      }
    }
  }
  return std::make_tuple(umin, usubmin, j1, j2);
}

template <typename idx>
always_inline std::tuple<double, double, idx, idx>
find_umins_avx2(
    idx dim, idx i, const double *restrict assign_cost,
    const double *restrict v) {
#ifdef __aarch64__
  // On ARM64, fall back to regular implementation
  return find_umins_regular(dim, i, assign_cost, v);
#else
  if (dim < DOUBLE_MIN_DIM) {
    return find_umins_regular(dim, i, assign_cost, v);
  }
  const double *local_cost = assign_cost + i * dim;
  __m256i idxvec = _mm256_setr_epi64x(0, 1, 2, 3);
  __m256i j1vec = _mm256_set1_epi64x(-1), j2vec = _mm256_set1_epi64x(-1);
  __m256d uminvec = _mm256_set1_pd(std::numeric_limits<double>::max()),
          usubminvec = _mm256_set1_pd(std::numeric_limits<double>::max());
  for (idx j = 0; j < dim - 3; j += 4) {
    __m256d acvec = _mm256_loadu_pd(local_cost + j);
    __m256d vvec = _mm256_loadu_pd(v + j);
    __m256d h = _mm256_sub_pd(acvec, vvec);
    __m256d cmp = _mm256_cmp_pd(h, uminvec, _CMP_LE_OQ);
    usubminvec = _mm256_blendv_pd(usubminvec, uminvec, cmp);
    j2vec = _mm256_blendv_epi8(
        j2vec, j1vec, _mm256_castpd_si256(cmp));
    uminvec = _mm256_blendv_pd(uminvec, h, cmp);
    j1vec = _mm256_blendv_epi8(
        j1vec, idxvec, _mm256_castpd_si256(cmp));
    cmp = _mm256_andnot_pd(cmp, _mm256_cmp_pd(h, usubminvec, _CMP_LT_OQ));
    usubminvec = _mm256_blendv_pd(usubminvec, h, cmp);
    j2vec = _mm256_blendv_epi8(
        j2vec, idxvec, _mm256_castpd_si256(cmp));
    idxvec = _mm256_add_epi64(idxvec, _mm256_set1_epi64x(4));
  }
  alignas(__m256d) double uminmem[4], usubminmem[4];
  alignas(__m256d) int64_t j1mem[4], j2mem[4];
  _mm256_store_pd(uminmem, uminvec);
  _mm256_store_pd(usubminmem, usubminvec);
  _mm256_store_si256(reinterpret_cast<__m256i*>(j1mem), j1vec);
  _mm256_store_si256(reinterpret_cast<__m256i*>(j2mem), j2vec);

  idx j1 = -1, j2 = -1;
  double umin = std::numeric_limits<double>::max(),
         usubmin = std::numeric_limits<double>::max();
  for (int vi = 0; vi < 4; vi++) {
    double h = uminmem[vi];
    if (h < usubmin) {
      idx jnew = j1mem[vi];
      if (h >= umin) {
        usubmin = h;
        j2 = jnew;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = jnew;
      }
    }
  }
  for (int vi = 0; vi < 4; vi++) {
    double h = usubminmem[vi];
    if (h < usubmin) {
      usubmin = h;
      j2 = j2mem[vi];
    }
  }
  for (idx j = dim & 0xFFFFFFFCu; j < dim; j++) {
    double h = local_cost[j] - v[j];
    if (h < usubmin) {
      if (h >= umin) {
        usubmin = h;
        j2 = j;
      } else {
        usubmin = umin;
        umin = h;
        j2 = j1;
        j1 = j;
      }
    }
  }
  return std::make_tuple(umin, usubmin, j1, j2);
#endif
}

// Update find_umins to use NEON when available
template <bool simd, typename idx, typename cost>
always_inline std::tuple<cost, cost, idx, idx>
find_umins(
    idx dim, idx i, const cost *restrict assign_cost,
    const cost *restrict v) {
  if constexpr(simd) {
    if constexpr(std::is_same_v<cost, float>) {
      #ifdef __aarch64__
        return find_umins_neon(dim, i, assign_cost, v);
      #else
        return find_umins_regular(dim, i, assign_cost, v);
      #endif
    } else if constexpr(std::is_same_v<cost, double>) {
      #ifdef __aarch64__
        return find_umins_neon_double(dim, i, assign_cost, v);
      #else
        return find_umins_avx2(dim, i, assign_cost, v);
      #endif
    }
  }
  return find_umins_regular(dim, i, assign_cost, v);
}

/// @brief Exact Jonker-Volgenant algorithm.
/// @param dim in problem size
/// @param assign_cost in cost matrix
/// @param verbose in indicates whether to report the progress to stdout
/// @param rowsol out column assigned to row in solution / size dim
/// @param colsol out row assigned to column in solution / size dim
/// @param u out dual variables, row reduction numbers / size dim
/// @param v out dual variables, column reduction numbers / size dim
/// @return achieved minimum assignment cost
template <bool avx2, bool verbose, typename idx, typename cost>
cost lap(int dim, const cost *restrict assign_cost,
         idx *restrict rowsol, idx *restrict colsol,
         cost *restrict u, cost *restrict v) {
  auto collist = std::make_unique<idx[]>(dim);  // list of columns to be scanned in various ways.
  auto matches = std::make_unique<idx[]>(dim);  // counts how many times a row could be assigned.
  auto d = std::make_unique<cost[]>(dim);       // 'cost-distance' in augmenting path calculation.
  auto pred = std::make_unique<idx[]>(dim);     // row-predecessor of column in augmenting/alternating path.

  // init how many times a row will be assigned in the column reduction.
  #if _OPENMP >= 201307
  #pragma omp simd
  #endif
  for (idx i = 0; i < dim; i++) {
    matches[i] = 0;
  }

  // COLUMN REDUCTION
  for (idx j = dim - 1; j >= 0; j--) {   // reverse order gives better results.
    // find minimum cost over rows.
    cost min = assign_cost[j];
    idx imin = 0;
    for (idx i = 1; i < dim; i++) {
      const cost *local_cost = &assign_cost[i * dim];
      if (local_cost[j] < min) {
        min = local_cost[j];
        imin = i;
      }
    }
    v[j] = min;

    if (++matches[imin] == 1) {
      // init assignment if minimum row assigned for first time.
      rowsol[imin] = j;
      colsol[j] = imin;
    } else {
      colsol[j] = -1;        // row already assigned, column not assigned.
    }
  }
  if (verbose) {
    printf("lapjv: COLUMN REDUCTION finished\n");
  }

  // REDUCTION TRANSFER
  auto free = matches.get();  // list of unassigned rows.
  idx numfree = 0;
  for (idx i = 0; i < dim; i++) {
    const cost *local_cost = &assign_cost[i * dim];
    if (matches[i] == 0) {  // fill list of unassigned 'free' rows.
      free[numfree++] = i;
    } else if (matches[i] == 1) {  // transfer reduction from rows that are assigned once.
      idx j1 = rowsol[i];
      cost min = std::numeric_limits<cost>::max();
      for (idx j = 0; j < dim; j++) {
        if (j != j1) {
          if (local_cost[j] - v[j] < min) {
            min = local_cost[j] - v[j];
          }
        }
      }
      v[j1] = v[j1] - min;
    }
  }
  if (verbose) {
    printf("lapjv: REDUCTION TRANSFER finished\n");
  }

  // AUGMENTING ROW REDUCTION
  for (int loopcnt = 0; loopcnt < 2; loopcnt++) {  // loop to be done twice.
    // scan all free rows.
    // in some cases, a free row may be replaced with another one to be scanned next.
    idx k = 0;
    idx prevnumfree = numfree;
    numfree = 0;  // start list of rows still free after augmenting row reduction.
    while (k < prevnumfree) {
      idx i = free[k++];

      // find minimum and second minimum reduced cost over columns.
      cost umin, usubmin;
      idx j1, j2;
      std::tie(umin, usubmin, j1, j2) = find_umins<avx2>(dim, i, assign_cost, v);

      idx i0 = colsol[j1];
      cost vj1_new = v[j1] - (usubmin - umin);
      bool vj1_lowers = vj1_new < v[j1];  // the trick to eliminate the epsilon bug
      if (vj1_lowers) {
        // change the reduction of the minimum column to increase the minimum
        // reduced cost in the row to the subminimum.
        v[j1] = vj1_new;
      } else if (i0 >= 0) {  // minimum and subminimum equal.
        // minimum column j1 is assigned.
        // swap columns j1 and j2, as j2 may be unassigned.
        j1 = j2;
        i0 = colsol[j2];
      }

      // (re-)assign i to j1, possibly de-assigning an i0.
      rowsol[i] = j1;
      colsol[j1] = i;

      if (i0 >= 0) {  // minimum column j1 assigned earlier.
        if (vj1_lowers) {
          // put in current k, and go back to that k.
          // continue augmenting path i - j1 with i0.
          free[--k] = i0;
        } else {
          // no further augmenting reduction possible.
          // store i0 in list of free rows for next phase.
          free[numfree++] = i0;
        }
      }
    }
    if (verbose) {
      printf("lapjv: AUGMENTING ROW REDUCTION %d / %d\n", loopcnt + 1, 2);
    }
  }  // for loopcnt

  // AUGMENT SOLUTION for each free row.
  for (idx f = 0; f < numfree; f++) {
    idx endofpath;
    idx freerow = free[f];       // start row of augmenting path.
    if (verbose) {
      printf("lapjv: AUGMENT SOLUTION row %d [%d / %d]\n",
             freerow, f + 1, numfree);
    }

    // Dijkstra shortest path algorithm.
    // runs until unassigned column added to shortest path tree.
    #if _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (idx j = 0; j < dim; j++) {
      d[j] = assign_cost[freerow * dim + j] - v[j];
      pred[j] = freerow;
      collist[j] = j;  // init column list.
    }

    idx low = 0; // columns in 0..low-1 are ready, now none.
    idx up = 0;  // columns in low..up-1 are to be scanned for current minimum, now none.
                 // columns in up..dim-1 are to be considered later to find new minimum,
                 // at this stage the list simply contains all columns
    bool unassigned_found = false;
    // initialized in the first iteration: low == up == 0
    idx last = 0;
    cost min = 0;
    do {
      if (up == low) {        // no more columns to be scanned for current minimum.
        last = low - 1;
        // scan columns for up..dim-1 to find all indices for which new minimum occurs.
        // store these indices between low..up-1 (increasing up).
        min = d[collist[up++]];
        for (idx k = up; k < dim; k++) {
          idx j = collist[k];
          cost h = d[j];
          if (h <= min) {
            if (h < min) {   // new minimum.
              up = low;      // restart list at index low.
              min = h;
            }
            // new index with same minimum, put on undex up, and extend list.
            collist[k] = collist[up];
            collist[up++] = j;
          }
        }

        // check if any of the minimum columns happens to be unassigned.
        // if so, we have an augmenting path right away.
        for (idx k = low; k < up; k++) {
          if (colsol[collist[k]] < 0) {
            endofpath = collist[k];
            unassigned_found = true;
            break;
          }
        }
      }

      if (!unassigned_found) {
        // update 'distances' between freerow and all unscanned columns, via next scanned column.
        idx j1 = collist[low];
        low++;
        idx i = colsol[j1];
        const cost *local_cost = &assign_cost[i * dim];
        cost h = local_cost[j1] - v[j1] - min;
        for (idx k = up; k < dim; k++) {
          idx j = collist[k];
          cost v2 = local_cost[j] - v[j] - h;
          if (v2 < d[j]) {
            pred[j] = i;
            if (v2 == min) {  // new column found at same minimum value
              if (colsol[j] < 0) {
                // if unassigned, shortest augmenting path is complete.
                endofpath = j;
                unassigned_found = true;
                break;
              } else {  // else add to list to be scanned right away.
                collist[k] = collist[up];
                collist[up++] = j;
              }
            }
            d[j] = v2;
          }
        }
      }
    } while (!unassigned_found);

    // update column prices.
    #if _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (idx k = 0; k <= last; k++) {
      idx j1 = collist[k];
      v[j1] = v[j1] + d[j1] - min;
    }

    // reset row and column assignments along the alternating path.
    {
      idx i;
      do {
        i = pred[endofpath];
        colsol[endofpath] = i;
        idx j1 = endofpath;
        endofpath = rowsol[i];
        rowsol[i] = j1;
      } while (i != freerow);
    }
  }
  if (verbose) {
    printf("lapjv: AUGMENT SOLUTION finished\n");
  }

  // calculate optimal cost.
  cost lapcost = 0;
  #if _OPENMP >= 201307
  #pragma omp simd reduction(+:lapcost)
  #endif
  for (idx i = 0; i < dim; i++) {
    const cost *local_cost = &assign_cost[i * dim];
    idx j = rowsol[i];
    u[i] = local_cost[j] - v[j];
    lapcost += local_cost[j];
  }
  if (verbose) {
    printf("lapjv: optimal cost calculated\n");
  }

  return lapcost;
}
