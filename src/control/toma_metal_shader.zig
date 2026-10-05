//! Exact F32 Metal primitives for staged ToMA parity.

pub const source: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup_matrix>
    \\using namespace metal;
    \\
    \\struct PatternParams {
    \\    uint source_tokens;
    \\    uint feature_width;
    \\    uint destination_tokens;
    \\    uint region_count;
    \\    uint region_size;
    \\    uint destinations_per_region;
    \\    uint token_side;
    \\    uint region_side;
    \\    uint selection_layout;
    \\    uint assignment_scope;
    \\    uint mask_selected;
    \\    uint raw_zero_norm;
    \\    float assignment_scale;
    \\};
    \\
    \\struct ApplyParams {
    \\    uint source_tokens;
    \\    uint destination_tokens;
    \\    uint width;
    \\    uint region_size;
    \\    uint destinations_per_region;
    \\    uint token_side;
    \\    uint region_side;
    \\    uint selection_layout;
    \\    uint assignment_scope;
    \\};
    \\
    \\struct CopyParams {
    \\    uint count;
    \\    uint source_offset;
    \\    uint destination_offset;
    \\};
    \\
    \\struct PositionParams {
    \\    uint image_source_tokens;
    \\    uint image_destination_tokens;
    \\    uint caption_tokens;
    \\};
    \\
    \\static inline uint region_token(
    \\    constant PatternParams& p,
    \\    uint region,
    \\    uint local
    \\) {
    \\    if (p.selection_layout != 1) return region * p.region_size + local;
    \\    uint tile_side = p.token_side / p.region_side;
    \\    uint region_y = region / p.region_side;
    \\    uint region_x = region - region_y * p.region_side;
    \\    uint local_y = local / tile_side;
    \\    uint local_x = local - local_y * tile_side;
    \\    return (region_y * tile_side + local_y) * p.token_side +
    \\        region_x * tile_side + local_x;
    \\}
    \\
    \\static inline uint source_region(constant PatternParams& p, uint source) {
    \\    if (p.selection_layout == 0) return 0;
    \\    if (p.selection_layout == 2) return source / p.region_size;
    \\    uint tile_side = p.token_side / p.region_side;
    \\    uint y = source / p.token_side;
    \\    uint x = source - y * p.token_side;
    \\    return (y / tile_side) * p.region_side + x / tile_side;
    \\}
    \\
    \\static inline uint apply_region_token(
    \\    constant ApplyParams& p,
    \\    uint region,
    \\    uint local
    \\) {
    \\    if (p.selection_layout != 1) return region * p.region_size + local;
    \\    uint tile_side = p.token_side / p.region_side;
    \\    uint region_y = region / p.region_side;
    \\    uint region_x = region - region_y * p.region_side;
    \\    uint local_y = local / tile_side;
    \\    uint local_x = local - local_y * tile_side;
    \\    return (region_y * tile_side + local_y) * p.token_side +
    \\        region_x * tile_side + local_x;
    \\}
    \\
    \\static inline uint apply_source_region(
    \\    constant ApplyParams& p,
    \\    uint source
    \\) {
    \\    if (p.selection_layout == 0) return 0;
    \\    if (p.selection_layout == 2) return source / p.region_size;
    \\    uint tile_side = p.token_side / p.region_side;
    \\    uint y = source / p.token_side;
    \\    uint x = source - y * p.token_side;
    \\    return (y / tile_side) * p.region_side + x / tile_side;
    \\}
    \\
    \\static inline float row_dot(
    \\    const device float* input,
    \\    uint width,
    \\    uint a,
    \\    uint b
    \\) {
    \\    float sum = 0.0f;
    \\    for (uint channel = 0; channel < width; channel++)
    \\        sum += input[a * width + channel] * input[b * width + channel];
    \\    return sum;
    \\}
    \\
    \\kernel void toma_normalize(
    \\    const device float* input [[buffer(0)]],
    \\    device float* output [[buffer(1)]],
    \\    constant PatternParams& p [[buffer(2)]],
    \\    uint token [[thread_position_in_grid]]
    \\) {
    \\    if (token >= p.source_tokens) return;
    \\    uint base = token * p.feature_width;
    \\    float norm_sq = 0.0f;
    \\    for (uint channel = 0; channel < p.feature_width; channel++) {
    \\        float value = input[base + channel];
    \\        norm_sq += value * value;
    \\    }
    \\    float norm = sqrt(norm_sq);
    \\    bool divide = p.raw_zero_norm != 0 || norm > 1.0e-12f;
    \\    for (uint channel = 0; channel < p.feature_width; channel++)
    \\        output[base + channel] = divide ? input[base + channel] / norm : 0.0f;
    \\}
    \\
    \\kernel void toma_similarity(
    \\    const device float* normalized [[buffer(0)]],
    \\    device float* similarity [[buffer(1)]],
    \\    constant PatternParams& p [[buffer(2)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint area = p.region_size * p.region_size;
    \\    if (index >= p.region_count * area) return;
    \\    uint region = index / area;
    \\    uint within = index - region * area;
    \\    uint row = within / p.region_size;
    \\    uint column = within - row * p.region_size;
    \\    uint a = region_token(p, region, row);
    \\    uint b = region_token(p, region, column);
    \\    similarity[index] = row_dot(normalized, p.feature_width, a, b);
    \\}
    \\
    \\kernel void toma_facility(
    \\    const device float* similarity [[buffer(0)]],
    \\    device uint* destinations [[buffer(1)]],
    \\    device float* max_similarity [[buffer(2)]],
    \\    device uchar* selected [[buffer(3)]],
    \\    constant PatternParams& p [[buffer(4)]],
    \\    uint region [[thread_position_in_grid]]
    \\) {
    \\    if (region >= p.region_count) return;
    \\    uint sim_base = region * p.region_size * p.region_size;
    \\    uint scratch_base = region * p.region_size;
    \\    for (uint index = 0; index < p.region_size; index++)
    \\        selected[scratch_base + index] = 0;
    \\    uint first = 0;
    \\    float best_score = -INFINITY;
    \\    for (uint candidate = 0; candidate < p.region_size; candidate++) {
    \\        float score = 0.0f;
    \\        for (uint source = 0; source < p.region_size; source++)
    \\            score += similarity[sim_base + candidate * p.region_size + source];
    \\        if (score > best_score) {
    \\            first = candidate;
    \\            best_score = score;
    \\        }
    \\    }
    \\    uint output_base = region * p.destinations_per_region;
    \\    destinations[output_base] = region_token(p, region, first);
    \\    selected[scratch_base + first] = 1;
    \\    for (uint source = 0; source < p.region_size; source++)
    \\        max_similarity[scratch_base + source] =
    \\            similarity[sim_base + first * p.region_size + source];
    \\    for (uint slot = 1; slot < p.destinations_per_region; slot++) {
    \\        uint next = 0;
    \\        float best_gain = -INFINITY;
    \\        for (uint candidate = 0; candidate < p.region_size; candidate++) {
    \\            if (p.mask_selected != 0 && selected[scratch_base + candidate] != 0)
    \\                continue;
    \\            float gain = 0.0f;
    \\            for (uint source = 0; source < p.region_size; source++) {
    \\                float value =
    \\                    similarity[sim_base + candidate * p.region_size + source];
    \\                gain += max(0.0f, value - max_similarity[scratch_base + source]);
    \\            }
    \\            if (gain > best_gain) {
    \\                next = candidate;
    \\                best_gain = gain;
    \\            }
    \\        }
    \\        destinations[output_base + slot] = region_token(p, region, next);
    \\        selected[scratch_base + next] = 1;
    \\        for (uint source = 0; source < p.region_size; source++) {
    \\            float value = similarity[sim_base + next * p.region_size + source];
    \\            max_similarity[scratch_base + source] =
    \\                max(max_similarity[scratch_base + source], value);
    \\        }
    \\    }
    \\}
    \\
    \\kernel void toma_clear(
    \\    device float* output [[buffer(0)]],
    \\    constant PatternParams& p [[buffer(1)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint count = p.destination_tokens * p.source_tokens;
    \\    if (index < count) output[index] = 0.0f;
    \\}
    \\
    \\kernel void toma_assignment(
    \\    const device float* normalized [[buffer(0)]],
    \\    const device uint* destinations [[buffer(1)]],
    \\    device float* assignment [[buffer(2)]],
    \\    constant PatternParams& p [[buffer(3)]],
    \\    uint source [[thread_position_in_grid]]
    \\) {
    \\    if (source >= p.source_tokens) return;
    \\    uint first = 0;
    \\    uint count = p.destination_tokens;
    \\    if (p.assignment_scope != 0) {
    \\        first = source_region(p, source) * p.destinations_per_region;
    \\        count = p.destinations_per_region;
    \\    }
    \\    float maximum = -INFINITY;
    \\    for (uint local = 0; local < count; local++) {
    \\        uint destination = first + local;
    \\        float score = p.assignment_scale * row_dot(
    \\            normalized, p.feature_width, destinations[destination], source);
    \\        maximum = max(maximum, score);
    \\    }
    \\    float denominator = 0.0f;
    \\    for (uint local = 0; local < count; local++) {
    \\        uint destination = first + local;
    \\        float score = p.assignment_scale * row_dot(
    \\            normalized, p.feature_width, destinations[destination], source);
    \\        float weight = exp(score - maximum);
    \\        assignment[destination * p.source_tokens + source] = weight;
    \\        denominator += weight;
    \\    }
    \\    for (uint local = 0; local < count; local++) {
    \\        uint destination = first + local;
    \\        assignment[destination * p.source_tokens + source] /= denominator;
    \\    }
    \\}
    \\
    \\kernel void toma_gather(
    \\    const device float* normalized [[buffer(0)]],
    \\    const device uint* destinations [[buffer(1)]],
    \\    device float* selected_features [[buffer(2)]],
    \\    constant PatternParams& p [[buffer(3)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint count = p.destination_tokens * p.feature_width;
    \\    if (index >= count) return;
    \\    uint destination = index / p.feature_width;
    \\    uint channel = index - destination * p.feature_width;
    \\    selected_features[index] =
    \\        normalized[destinations[destination] * p.feature_width + channel];
    \\}
    \\
    \\kernel void toma_scores_mma(
    \\    const device float* selected_features [[buffer(0)]],
    \\    const device float* normalized [[buffer(1)]],
    \\    device float* scores [[buffer(2)]],
    \\    constant PatternParams& p [[buffer(3)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint tile_columns = (p.source_tokens + 31) / 32;
    \\    uint tile_row = group / tile_columns;
    \\    uint tile_column = group - tile_row * tile_columns;
    \\    uint row_base = tile_row * 32;
    \\    uint column_base = tile_column * 32;
    \\    if (row_base + 32 > p.destination_tokens ||
    \\        column_base + 32 > p.source_tokens) {
    \\        uint row = row_base + tid;
    \\        if (row < p.destination_tokens) {
    \\            for (uint local_column = 0; local_column < 32; local_column++) {
    \\                uint column = column_base + local_column;
    \\                if (column >= p.source_tokens) continue;
    \\                float sum = 0.0f;
    \\                for (uint feature = 0; feature < p.feature_width; feature++)
    \\                    sum += selected_features[row * p.feature_width + feature] *
    \\                        normalized[column * p.feature_width + feature];
    \\                scores[row * p.source_tokens + column] = sum;
    \\            }
    \\        }
    \\        return;
    \\    }
    \\    simdgroup_float8x8 accumulators[4][4];
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            accumulators[row][column] =
    \\                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    for (uint feature = 0; feature < p.feature_width; feature += 8) {
    \\        simdgroup_float8x8 left[4];
    \\        simdgroup_float8x8 right[4];
    \\        for (uint row = 0; row < 4; row++)
    \\            simdgroup_load(
    \\                left[row],
    \\                selected_features +
    \\                    (row_base + row * 8) * p.feature_width + feature,
    \\                p.feature_width);
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_load(
    \\                right[column],
    \\                normalized +
    \\                    (column_base + column * 8) * p.feature_width + feature,
    \\                p.feature_width,
    \\                ulong2(0, 0),
    \\                true);
    \\        for (uint row = 0; row < 4; row++)
    \\            for (uint column = 0; column < 4; column++)
    \\                simdgroup_multiply_accumulate(
    \\                    accumulators[row][column],
    \\                    left[row],
    \\                    right[column],
    \\                    accumulators[row][column]);
    \\    }
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_store(
    \\                accumulators[row][column],
    \\                scores + (row_base + row * 8) * p.source_tokens +
    \\                    column_base + column * 8,
    \\                p.source_tokens);
    \\}
    \\
    \\kernel void toma_score_softmax(
    \\    device float* scores [[buffer(0)]],
    \\    constant PatternParams& p [[buffer(1)]],
    \\    uint source [[thread_position_in_grid]]
    \\) {
    \\    if (source >= p.source_tokens) return;
    \\    float maximum = -INFINITY;
    \\    for (uint destination = 0; destination < p.destination_tokens; destination++)
    \\        maximum = max(
    \\            maximum,
    \\            p.assignment_scale * scores[destination * p.source_tokens + source]);
    \\    float denominator = 0.0f;
    \\    for (uint destination = 0; destination < p.destination_tokens; destination++) {
    \\        uint index = destination * p.source_tokens + source;
    \\        float weight = exp(p.assignment_scale * scores[index] - maximum);
    \\        scores[index] = weight;
    \\        denominator += weight;
    \\    }
    \\    for (uint destination = 0; destination < p.destination_tokens; destination++)
    \\        scores[destination * p.source_tokens + source] /= denominator;
    \\}
    \\
    \\kernel void toma_row_norm(
    \\    const device float* assignment [[buffer(0)]],
    \\    device float* merge_matrix [[buffer(1)]],
    \\    constant PatternParams& p [[buffer(2)]],
    \\    uint destination [[thread_position_in_grid]]
    \\) {
    \\    if (destination >= p.destination_tokens) return;
    \\    uint base = destination * p.source_tokens;
    \\    float mass = 0.0f;
    \\    for (uint source = 0; source < p.source_tokens; source++)
    \\        mass += assignment[base + source];
    \\    for (uint source = 0; source < p.source_tokens; source++)
    \\        merge_matrix[base + source] = assignment[base + source] / mass;
    \\}
    \\
    \\kernel void toma_merge(
    \\    const device float* matrix [[buffer(0)]],
    \\    const device float* input [[buffer(1)]],
    \\    device float* output [[buffer(2)]],
    \\    constant ApplyParams& p [[buffer(3)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint count = p.destination_tokens * p.width;
    \\    if (index >= count) return;
    \\    uint destination = index / p.width;
    \\    uint channel = index - destination * p.width;
    \\    float sum = 0.0f;
    \\    if (p.assignment_scope == 0) {
    \\        for (uint source = 0; source < p.source_tokens; source++)
    \\            sum += matrix[destination * p.source_tokens + source] *
    \\                input[source * p.width + channel];
    \\    } else {
    \\        uint region = destination / p.destinations_per_region;
    \\        for (uint local = 0; local < p.region_size; local++) {
    \\            uint source = apply_region_token(p, region, local);
    \\            sum += matrix[destination * p.source_tokens + source] *
    \\                input[source * p.width + channel];
    \\        }
    \\    }
    \\    output[index] = sum;
    \\}
    \\
    \\kernel void toma_unmerge(
    \\    const device float* matrix [[buffer(0)]],
    \\    const device float* input [[buffer(1)]],
    \\    device float* output [[buffer(2)]],
    \\    constant ApplyParams& p [[buffer(3)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint count = p.source_tokens * p.width;
    \\    if (index >= count) return;
    \\    uint source = index / p.width;
    \\    uint channel = index - source * p.width;
    \\    float sum = 0.0f;
    \\    if (p.assignment_scope == 0) {
    \\        for (uint destination = 0; destination < p.destination_tokens; destination++)
    \\            sum += matrix[destination * p.source_tokens + source] *
    \\                input[destination * p.width + channel];
    \\    } else {
    \\        uint region = apply_source_region(p, source);
    \\        uint first = region * p.destinations_per_region;
    \\        for (uint local = 0; local < p.destinations_per_region; local++) {
    \\            uint destination = first + local;
    \\            sum += matrix[destination * p.source_tokens + source] *
    \\                input[destination * p.width + channel];
    \\        }
    \\    }
    \\    output[index] = sum;
    \\}
    \\
    \\kernel void toma_merge_mma(
    \\    const device float* matrix [[buffer(0)]],
    \\    const device float* input [[buffer(1)]],
    \\    device float* output [[buffer(2)]],
    \\    constant ApplyParams& p [[buffer(3)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint tile_columns = (p.width + 31) / 32;
    \\    uint tile_row = group / tile_columns;
    \\    uint tile_column = group - tile_row * tile_columns;
    \\    uint row_base = tile_row * 32;
    \\    uint column_base = tile_column * 32;
    \\    if (row_base + 32 > p.destination_tokens || column_base + 32 > p.width) {
    \\        uint row = row_base + tid;
    \\        if (row < p.destination_tokens) {
    \\            for (uint local_column = 0; local_column < 32; local_column++) {
    \\                uint column = column_base + local_column;
    \\                if (column >= p.width) continue;
    \\                float sum = 0.0f;
    \\                for (uint source = 0; source < p.source_tokens; source++)
    \\                    sum += matrix[row * p.source_tokens + source] *
    \\                        input[source * p.width + column];
    \\                output[row * p.width + column] = sum;
    \\            }
    \\        }
    \\        return;
    \\    }
    \\    simdgroup_float8x8 accumulators[4][4];
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            accumulators[row][column] =
    \\                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    for (uint source = 0; source < p.source_tokens; source += 8) {
    \\        simdgroup_float8x8 left[4];
    \\        simdgroup_float8x8 right[4];
    \\        for (uint row = 0; row < 4; row++)
    \\            simdgroup_load(
    \\                left[row],
    \\                matrix + (row_base + row * 8) * p.source_tokens + source,
    \\                p.source_tokens);
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_load(
    \\                right[column],
    \\                input + source * p.width + column_base + column * 8,
    \\                p.width);
    \\        for (uint row = 0; row < 4; row++)
    \\            for (uint column = 0; column < 4; column++)
    \\                simdgroup_multiply_accumulate(
    \\                    accumulators[row][column],
    \\                    left[row],
    \\                    right[column],
    \\                    accumulators[row][column]);
    \\    }
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_store(
    \\                accumulators[row][column],
    \\                output + (row_base + row * 8) * p.width +
    \\                    column_base + column * 8,
    \\                p.width);
    \\}
    \\
    \\kernel void toma_unmerge_mma(
    \\    const device float* matrix [[buffer(0)]],
    \\    const device float* input [[buffer(1)]],
    \\    device float* output [[buffer(2)]],
    \\    constant ApplyParams& p [[buffer(3)]],
    \\    uint group [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]]
    \\) {
    \\    uint tile_columns = (p.width + 31) / 32;
    \\    uint tile_row = group / tile_columns;
    \\    uint tile_column = group - tile_row * tile_columns;
    \\    uint row_base = tile_row * 32;
    \\    uint column_base = tile_column * 32;
    \\    if (row_base + 32 > p.source_tokens || column_base + 32 > p.width) {
    \\        uint row = row_base + tid;
    \\        if (row < p.source_tokens) {
    \\            for (uint local_column = 0; local_column < 32; local_column++) {
    \\                uint column = column_base + local_column;
    \\                if (column >= p.width) continue;
    \\                float sum = 0.0f;
    \\                for (uint destination = 0; destination < p.destination_tokens;
    \\                     destination++)
    \\                    sum += matrix[destination * p.source_tokens + row] *
    \\                        input[destination * p.width + column];
    \\                output[row * p.width + column] = sum;
    \\            }
    \\        }
    \\        return;
    \\    }
    \\    simdgroup_float8x8 accumulators[4][4];
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            accumulators[row][column] =
    \\                make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\    for (uint destination = 0; destination < p.destination_tokens;
    \\         destination += 8) {
    \\        simdgroup_float8x8 left[4];
    \\        simdgroup_float8x8 right[4];
    \\        for (uint row = 0; row < 4; row++)
    \\            simdgroup_load(
    \\                left[row],
    \\                matrix + destination * p.source_tokens + row_base + row * 8,
    \\                p.source_tokens,
    \\                ulong2(0, 0),
    \\                true);
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_load(
    \\                right[column],
    \\                input + destination * p.width + column_base + column * 8,
    \\                p.width);
    \\        for (uint row = 0; row < 4; row++)
    \\            for (uint column = 0; column < 4; column++)
    \\                simdgroup_multiply_accumulate(
    \\                    accumulators[row][column],
    \\                    left[row],
    \\                    right[column],
    \\                    accumulators[row][column]);
    \\    }
    \\    for (uint row = 0; row < 4; row++)
    \\        for (uint column = 0; column < 4; column++)
    \\            simdgroup_store(
    \\                accumulators[row][column],
    \\                output + (row_base + row * 8) * p.width +
    \\                    column_base + column * 8,
    \\                p.width);
    \\}
    \\
    \\kernel void toma_copy_u32(
    \\    const device uint* input [[buffer(0)]],
    \\    device uint* output [[buffer(1)]],
    \\    constant CopyParams& p [[buffer(2)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    if (index >= p.count) return;
    \\    output[p.destination_offset + index] = input[p.source_offset + index];
    \\}
    \\
    \\kernel void toma_gather_positions(
    \\    const device ulong* full [[buffer(0)]],
    \\    const device uint* destinations [[buffer(1)]],
    \\    device ulong* reduced [[buffer(2)]],
    \\    constant PositionParams& p [[buffer(3)]],
    \\    uint index [[thread_position_in_grid]]
    \\) {
    \\    uint token_count = p.image_destination_tokens + p.caption_tokens;
    \\    if (index >= token_count * 3) return;
    \\    uint token = index / 3;
    \\    uint axis = index - token * 3;
    \\    uint source = token < p.image_destination_tokens
    \\        ? destinations[token]
    \\        : p.image_source_tokens + token - p.image_destination_tokens;
    \\    reduced[index] = full[source * 3 + axis];
    \\}
;
