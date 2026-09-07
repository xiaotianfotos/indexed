// SPDX-License-Identifier: Apache-2.0
#ifndef INDEXED_PRIVATE_ANE_BRIDGE_H
#define INDEXED_PRIVATE_ANE_BRIDGE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct IndexedANEProgram IndexedANEProgram;

// Fixed-shape synchronous bridge. Callers serialize evaluation and destruction.
// This experimental API uses Apple's private runtime, not public Core ML.
IndexedANEProgram *indexed_ane_create(
    const uint8_t *mil, size_t mil_bytes,
    const uint8_t *weight_data, size_t weight_data_bytes,
    const uint8_t *weight_scales, size_t weight_scales_bytes,
    const size_t *input_sizes, size_t input_count,
    const size_t *output_sizes, size_t output_count,
    char *error, size_t error_capacity);
// Share only equal-sized I/O surfaces; weights and compiled models stay distinct.
// The caller must serialize all programs in this sharing group through readback.
IndexedANEProgram *indexed_ane_create_sharing(
    const uint8_t *mil, size_t mil_bytes,
    const uint8_t *weight_data, size_t weight_data_bytes,
    const uint8_t *weight_scales, size_t weight_scales_bytes,
    const size_t *input_sizes, size_t input_count,
    const size_t *output_sizes, size_t output_count,
    IndexedANEProgram *sharing,
    char *error, size_t error_capacity);
bool indexed_ane_write(IndexedANEProgram *, size_t index, const void *bytes, size_t count, char *error, size_t error_capacity);
bool indexed_ane_evaluate(IndexedANEProgram *, char *error, size_t error_capacity);
bool indexed_ane_read(IndexedANEProgram *, size_t index, void *bytes, size_t count, char *error, size_t error_capacity);
// Borrow a read-only surface under the caller's sharing-group lock. The program
// must outlive the borrow; join all consumers before unlock or another evaluate.
void *indexed_ane_lock_output(IndexedANEProgram *, size_t index, size_t count, char *error, size_t error_capacity);
void indexed_ane_unlock_output(IndexedANEProgram *, size_t index);
// Independent storage ownership for a cached GPU mapping; retaining a surface
// must not keep its compiled model alive or form a model/workspace cycle.
void *indexed_ane_retain_output_surface(IndexedANEProgram *, size_t index);
void indexed_ane_release_surface(void *surface);
void indexed_ane_free(IndexedANEProgram *);
// Version-locked MLX JIT sources, already linked in the helper; no disk lookup.
const char *indexed_mlx_affine_kernel_header(void);
#ifdef __cplusplus
}
#endif
#endif
