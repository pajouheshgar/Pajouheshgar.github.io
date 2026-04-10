// Apply permutation to scatter positions and types into binned order.
// Matches CUDA apply_permutation (lenia_cuda.cu:110-129).

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

@group(0) @binding(0) var<storage, read> positions_in: array<vec2f>;
@group(0) @binding(1) var<storage, read> types_in: array<u32>;
@group(0) @binding(2) var<storage, read> permutation: array<u32>;
@group(0) @binding(3) var<storage, read_write> positions_out: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> types_out: array<u32>;
@group(0) @binding(5) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let dest = permutation[i];
    positions_out[dest] = positions_in[i];
    types_out[dest] = types_in[i];
}
