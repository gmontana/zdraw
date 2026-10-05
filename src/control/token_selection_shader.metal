#include <metal_stdlib>
using namespace metal;

struct TokenSelectionParams {
    uint source_tokens;
    uint feature_width;
    uint selected_tokens;
    float reconstruction_weight;
};

kernel void zdraw_token_score_zero(
    device float* scores [[buffer(0)]],
    constant TokenSelectionParams& p [[buffer(1)]],
    uint token [[thread_position_in_grid]]
) {
    if (token < p.source_tokens) scores[token] = 0.0f;
}

kernel void zdraw_token_score_mean(
    const device float* hidden [[buffer(0)]],
    device float* scores [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= p.source_tokens) return;
    const uint base = token * p.feature_width;
    float sum = 0.0f;
    for (uint feature = 0; feature < p.feature_width; ++feature)
        sum += hidden[base + feature];
    float score = sum / float(p.feature_width);
    if (isnan(score)) score = -INFINITY;
    if (score == 0.0f) score = 0.0f;
    scores[token] = score;
}

inline ulong zdraw_token_key(float score, uint token) {
    const uint bits = as_type<uint>(score);
    const uint ordered = (bits & 0x80000000u) != 0
        ? ~bits
        : bits ^ 0x80000000u;
    return (ulong(ordered) << 32) | ulong(token);
}

kernel void zdraw_token_select_topk(
    const device float* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lane_count [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {
    if (group != 0) return;
    threadgroup atomic_uint one_count;
    threadgroup ulong prefix;
    threadgroup ulong prefix_mask;
    threadgroup uint remaining;
    threadgroup uint offsets[256];

    if (lane == 0) {
        prefix = 0;
        prefix_mask = 0;
        remaining = p.selected_tokens;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int bit = 63; bit >= 0; --bit) {
        const ulong bit_mask = 1ul << uint(bit);
        if (lane == 0)
            atomic_store_explicit(
                &one_count,
                0,
                memory_order_relaxed
            );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint local_count = 0;
        for (
            uint token = lane;
            token < p.source_tokens;
            token += lane_count
        ) {
            const ulong key = zdraw_token_key(scores[token], token);
            if (
                (key & prefix_mask) == prefix &&
                (key & bit_mask) != 0
            ) {
                ++local_count;
            }
        }
        atomic_fetch_add_explicit(
            &one_count,
            local_count,
            memory_order_relaxed
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            const uint ones = atomic_load_explicit(
                &one_count,
                memory_order_relaxed
            );
            if (remaining <= ones) {
                prefix |= bit_mask;
            } else {
                remaining -= ones;
            }
            prefix_mask |= bit_mask;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint start = uint(
        (ulong(p.source_tokens) * ulong(lane)) / ulong(lane_count)
    );
    const uint end = uint(
        (ulong(p.source_tokens) * ulong(lane + 1)) / ulong(lane_count)
    );
    uint selected = 0;
    for (uint token = start; token < end; ++token) {
        if (zdraw_token_key(scores[token], token) >= prefix)
            ++selected;
    }
    offsets[lane] = selected;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        uint offset = 0;
        for (uint index = 0; index < lane_count; ++index) {
            const uint count = offsets[index];
            offsets[index] = offset;
            offset += count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint output = offsets[lane];
    for (uint token = start; token < end; ++token) {
        if (zdraw_token_key(scores[token], token) >= prefix)
            indices[output++] = token;
    }
}

kernel void zdraw_token_gather_state(
    const device float* full [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    device float* compact [[buffer(2)]],
    constant TokenSelectionParams& p [[buffer(3)]],
    uint element [[thread_position_in_grid]]
) {
    const uint count = p.selected_tokens * p.feature_width;
    if (element >= count) return;
    const uint row = element / p.feature_width;
    const uint feature = element - row * p.feature_width;
    compact[element] = full[indices[row] * p.feature_width + feature];
}

kernel void zdraw_token_gather_positions(
    const device uint* full [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    device uint* compact [[buffer(2)]],
    constant TokenSelectionParams& p [[buffer(3)]],
    uint element [[thread_position_in_grid]]
) {
    if (element >= p.selected_tokens * 3) return;
    const uint row = element / 3;
    const uint axis = element - row * 3;
    compact[element] = full[indices[row] * 3 + axis];
}

kernel void zdraw_token_scatter_state(
    const device float* compact [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    device float* full [[buffer(2)]],
    constant TokenSelectionParams& p [[buffer(3)]],
    uint element [[thread_position_in_grid]]
) {
    const uint count = p.selected_tokens * p.feature_width;
    if (element >= count) return;
    const uint row = element / p.feature_width;
    const uint feature = element - row * p.feature_width;
    full[indices[row] * p.feature_width + feature] = compact[element];
}

inline void zdraw_normalize_row(
    const device float* input,
    device float* output,
    uint row,
    uint width
) {
    const ulong base = ulong(row) * ulong(width);
    float squared = 0.0f;
    for (uint feature = 0; feature < width; ++feature) {
        const float value = input[base + feature];
        squared += value * value;
    }
    const float inverse = rsqrt(max(squared, 1.0e-24f));
    for (uint feature = 0; feature < width; ++feature)
        output[base + feature] = input[base + feature] * inverse;
}

kernel void zdraw_token_normalize_source(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < p.source_tokens)
        zdraw_normalize_row(input, output, row, p.feature_width);
}

kernel void zdraw_token_normalize_destination(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < p.selected_tokens)
        zdraw_normalize_row(input, output, row, p.feature_width);
}

kernel void zdraw_token_assign_max(
    const device float* scores [[buffer(0)]],
    device uint* assignments [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint source [[thread_position_in_grid]]
) {
    if (source >= p.source_tokens) return;
    const ulong base = ulong(source) * ulong(p.selected_tokens);
    float best_score = -INFINITY;
    uint best_destination = 0;
    for (uint destination = 0; destination < p.selected_tokens; ++destination) {
        const float score = scores[base + destination];
        if (score > best_score) {
            best_score = score;
            best_destination = destination;
        }
    }
    assignments[source] = best_destination;
}

kernel void zdraw_token_clear_inverse(
    device uint* inverse [[buffer(0)]],
    constant TokenSelectionParams& p [[buffer(1)]],
    uint token [[thread_position_in_grid]]
) {
    if (token < p.source_tokens) inverse[token] = UINT_MAX;
}

kernel void zdraw_token_mark_inverse(
    const device uint* indices [[buffer(0)]],
    device uint* inverse [[buffer(1)]],
    constant TokenSelectionParams& p [[buffer(2)]],
    uint compact_row [[thread_position_in_grid]]
) {
    if (compact_row < p.selected_tokens)
        inverse[indices[compact_row]] = compact_row;
}

kernel void zdraw_token_reconstruct_cosine_residual(
    const device float* compact [[buffer(0)]],
    const device uint* assignments [[buffer(1)]],
    const device uint* inverse [[buffer(2)]],
    device float* full [[buffer(3)]],
    constant TokenSelectionParams& p [[buffer(4)]],
    uint element [[thread_position_in_grid]]
) {
    const ulong count = ulong(p.source_tokens) * ulong(p.feature_width);
    if (ulong(element) >= count) return;
    const uint token = element / p.feature_width;
    const uint feature = element - token * p.feature_width;
    const uint selected_row = inverse[token];
    if (selected_row != UINT_MAX) {
        full[element] =
            compact[ulong(selected_row) * ulong(p.feature_width) + feature];
        return;
    }
    const uint destination = assignments[token];
    const float expanded =
        compact[ulong(destination) * ulong(p.feature_width) + feature];
    const float weight = p.reconstruction_weight;
    full[element] = (1.0f - weight) * expanded + weight * full[element];
}
