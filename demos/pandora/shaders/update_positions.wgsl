// Update positions: x -= grad_E * dt, then wrap to periodic domain.
// Also integrates the state vector: s = normalize(s + state_force * dt).

const MAX_STATE_DIM: u32 = 16u;

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
    state_dim: u32,
    freeze_states: u32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> grad_e: array<vec2f>;
@group(0) @binding(2) var<uniform> params: SimParams;
@group(0) @binding(3) var<storage, read_write> states: array<f32>;
@group(0) @binding(4) var<storage, read> state_force: array<f32>;

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

    // State update: s = normalize(s + dt * state_force)
    if (params.freeze_states != 0u) { return; }
    let D = min(params.state_dim, MAX_STATE_DIM);
    var s_new: array<f32, MAX_STATE_DIM>;
    var len_sq: f32 = 0.0;
    for (var d = 0u; d < D; d++) {
        s_new[d] = states[i * D + d] + state_force[i * D + d] * params.dt;
        len_sq += s_new[d] * s_new[d];
    }
    // Keep the previous (unit) state if the update degenerates to ~zero length
    if (len_sq > 1e-12) {
        let inv_len = inverseSqrt(len_sq);
        for (var d = 0u; d < D; d++) {
            states[i * D + d] = s_new[d] * inv_len;
        }
    }
}
