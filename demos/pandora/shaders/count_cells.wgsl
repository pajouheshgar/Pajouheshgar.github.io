// Count particles per hash grid cell.
// Matches CUDA grid_count_particles (lenia_cuda.cu:54-77).

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

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

fn dilate2d(x_in: u32) -> u32 {
    var x = x_in & 0x3FFu;
    x = (x | (x << 16u)) & 0x0000FFFFu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

fn pos2cell(pos: vec2f) -> vec2u {
    let cx = u32(i32(floor(pos.x * params.inv_eps)) + i32(params.grid_size / 2u)) & (params.grid_size - 1u);
    let cy = u32(i32(floor(pos.y * params.inv_eps)) + i32(params.grid_size / 2u)) & (params.grid_size - 1u);
    return vec2u(cx, cy);
}

fn cell2hash(cell: vec2u) -> u32 {
    return dilate2d(cell.x) | (dilate2d(cell.y) << 1u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let pos = positions[i];
    let cell = pos2cell(pos);
    let hash = cell2hash(cell);

    atomicAdd(&cell_counts[hash], 1u);
}
