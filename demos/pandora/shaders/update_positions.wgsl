// Update positions: x -= grad_E * dt, then wrap to periodic domain.

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

@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> grad_e: array<vec2f>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    var pos = positions[i] - grad_e[i] * params.dt;

    // Wrap to [-half_box, half_box)
    let L = params.box_size;
    let hL = params.half_box;
    pos.x = (pos.x + hL) % L;
    if (pos.x < 0.0) { pos.x += L; }
    pos.x -= hL;

    pos.y = (pos.y + hL) % L;
    if (pos.y < 0.0) { pos.y += L; }
    pos.y -= hL;

    positions[i] = pos;
}
