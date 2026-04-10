// Exclusive prefix sum (Blelloch scan) for bin_offsets.
// Converts cell_counts -> bin_offsets.
// Supports up to 256*256 = 65536 cells with a single workgroup of 256 threads.
// For cell_count <= 65536, uses a sequential per-thread scan approach.

struct SimParams {
    N: u32,
    cell_count: u32,
    num_species: u32,
    grid_size: u32,
    log2_grid_size: u32,
    cell_radius: i32,
    box_size: f32,
    half_box: f32,
    eps: f32,
    inv_eps: f32,
    R_max_sq: f32,
    dt: f32,
};

@group(0) @binding(0) var<storage, read> cell_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> bin_offsets: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

// Shared memory for workgroup-level scan
var<workgroup> shared_data: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
    let tid = lid.x;
    let n = params.cell_count;
    let items_per_thread = (n + 255u) / 256u;

    // Phase 1: Each thread computes partial sum for its chunk
    var local_sum = 0u;
    let start = tid * items_per_thread;
    for (var k = 0u; k < items_per_thread; k++) {
        let idx = start + k;
        if (idx < n) {
            local_sum += cell_counts[idx];
        }
    }
    shared_data[tid] = local_sum;
    workgroupBarrier();

    // Phase 2: Blelloch scan on the 256 partial sums
    // Up-sweep
    var offset = 1u;
    for (var d = 128u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            if (bi < 256u) {
                shared_data[bi] += shared_data[ai];
            }
        }
        offset <<= 1u;
    }

    // Set last element to 0 (exclusive scan)
    if (tid == 0u) {
        shared_data[255] = 0u;
    }

    // Down-sweep
    for (var d = 1u; d < 256u; d <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            if (bi < 256u) {
                let tmp = shared_data[ai];
                shared_data[ai] = shared_data[bi];
                shared_data[bi] += tmp;
            }
        }
    }
    workgroupBarrier();

    // Phase 3: Each thread writes bin_offsets for its chunk using the scanned prefix
    let prefix = shared_data[tid];
    var running = prefix;
    for (var k = 0u; k < items_per_thread; k++) {
        let idx = start + k;
        if (idx < n) {
            bin_offsets[idx] = running;
            running += cell_counts[idx];
        }
    }

    // Thread with the last chunk writes the total at bin_offsets[n]
    // The last thread's running sum after its chunk = prefix + local_sum for its chunk
    workgroupBarrier();
    if (tid == 255u) {
        bin_offsets[n] = running;
    }
}
