// Particle-centric forward kernel with hash grid neighbor search.
// Pairwise U interactions are weighted by relu(dot(s_i, s_j)) where s is a
// unit-length state vector of dimension state_dim (<= MAX_STATE_DIM).

const KERNEL_SIGMA_CUTOFF: f32 = 3.0;
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

struct SpeciesParams {
    mu_k: f32,
    sigma_k: f32,
    w_k: f32,
    mu_g: f32,
    sigma_g: f32,
    c_rep: f32,
    r_min_k: f32,
    r_max_k: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> types: array<u32>;
@group(0) @binding(2) var<storage, read> bin_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> species: array<SpeciesParams>;
@group(0) @binding(4) var<uniform> params: SimParams;
@group(0) @binding(5) var<storage, read_write> scalar_fields: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> grad_e_out: array<vec2f>;
@group(0) @binding(7) var<storage, read> states: array<f32>;
@group(0) @binding(8) var<storage, read_write> state_force_out: array<f32>;

fn dilate2d(x_in: u32) -> u32 {
    var x = x_in & 0x3FFu;
    x = (x | (x << 16u)) & 0x0000FFFFu;
    x = (x | (x << 8u)) & 0x00FF00FFu;
    x = (x | (x << 4u)) & 0x0F0F0F0Fu;
    x = (x | (x << 2u)) & 0x33333333u;
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

fn pos2cell(pos: vec2f) -> vec2i {
    let cx = i32(floor(pos.x * params.inv_eps)) + i32(params.grid_size / 2u);
    let cy = i32(floor(pos.y * params.inv_eps)) + i32(params.grid_size / 2u);
    return vec2i(cx, cy);
}

fn cell2hash(cx: u32, cy: u32) -> u32 {
    return dilate2d(cx) | (dilate2d(cy) << 1u);
}

fn wrap_periodic(d: f32, half_L: f32, L: f32) -> f32 {
    var r = d;
    if (r > half_L) { r -= L; }
    else if (r < -half_L) { r += L; }
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }

    let p_i = positions[i];
    let type_i = types[i];
    let sp_i = species[type_i];
    let gs = i32(params.grid_size);
    let mask = params.grid_size - 1u;
    let D = min(params.state_dim, MAX_STATE_DIM);

    // Load own state vector
    var s_i: array<f32, MAX_STATE_DIM>;
    var s_grad: array<f32, MAX_STATE_DIM>;
    for (var d = 0u; d < D; d++) {
        s_i[d] = states[i * D + d];
        s_grad[d] = 0.0;
    }

    // Find cell of particle i
    let cell_i = pos2cell(p_i);

    // Accumulators. U starts with the self-term (state weight = dot(s,s) = 1).
    let t_self = sp_i.mu_k / sp_i.sigma_k;
    var U_accum: f32 = sp_i.w_k * exp(-t_self * t_self);
    var R_accum: f32 = 0.0;
    var grad_U: vec2f = vec2f(0.0, 0.0);
    var grad_R: vec2f = vec2f(0.0, 0.0);

    let cr = params.cell_radius;

    // Iterate over neighbor cells
    for (var dy: i32 = -cr; dy <= cr; dy++) {
        let ny = u32((cell_i.y + dy) & i32(mask));
        for (var dx: i32 = -cr; dx <= cr; dx++) {
            let nx = u32((cell_i.x + dx) & i32(mask));
            let hash = cell2hash(nx, ny);

            let start = bin_offsets[hash];
            let end = bin_offsets[hash + 1u];

            // Process all particles in this neighbor cell
            for (var j: u32 = start; j < end; j++) {
                if (j == i) { continue; }

                let p_j = positions[j];
                let type_j = types[j];

                // Periodic displacement
                let dx_p = wrap_periodic(p_j.x - p_i.x, params.half_box, params.box_size);
                let dy_p = wrap_periodic(p_j.y - p_i.y, params.half_box, params.box_size);
                let dr = vec2f(dx_p, dy_p);

                let r_sq = dot(dr, dr);
                if (r_sq >= params.R_max_sq) { continue; }

                let inv_r = inverseSqrt(r_sq);
                let r = r_sq * inv_r;
                if (r < 1e-10) { continue; }

                let sp_j = species[type_j];

                // U-field contribution, weighted by relu(dot(s_i, s_j))
                if (r >= sp_j.r_min_k && r <= sp_j.r_max_k) {
                    var sw: f32 = 0.0;
                    for (var d = 0u; d < D; d++) {
                        sw += s_i[d] * states[j * D + d];
                    }
                    let state_w = max(sw, 0.0);

                    let t = (r - sp_j.mu_k) / sp_j.sigma_k;
                    let pf = exp(-t * t);
                    let dpf_dr = pf * (-2.0 * t / sp_j.sigma_k);

                    U_accum += sp_j.w_k * pf * state_w;

                    let factor = sp_j.w_k * dpf_dr * inv_r * state_w;
                    grad_U -= factor * dr;

                    if (sw > 0.0) {
                        let kw = sp_j.w_k * pf;
                        for (var d = 0u; d < D; d++) {
                            s_grad[d] += kw * states[j * D + d];
                        }
                    }
                }

                // Repulsion contribution (not state-weighted)
                if (r < 1.0) {
                    let omr = 1.0 - r;
                    R_accum += 0.5 * sp_j.c_rep * omr * omr;

                    let rep_factor = sp_j.c_rep * omr * inv_r;
                    grad_R += rep_factor * dr;
                }
            }
        }
    }

    // Growth function using particle i's species params
    let t_g = (U_accum - sp_i.mu_g) / sp_i.sigma_g;
    let G_val = exp(-t_g * t_g);
    let dG_dU = G_val * (-2.0 * t_g / sp_i.sigma_g);

    // Energy
    let E_val = R_accum - G_val;
    let grad_E = grad_R - dG_dU * grad_U;

    // State force: dG/dU times the component of s_grad tangent to the sphere at s_i
    var s_dot_grad: f32 = 0.0;
    for (var d = 0u; d < D; d++) {
        s_dot_grad += s_i[d] * s_grad[d];
    }
    for (var d = 0u; d < D; d++) {
        state_force_out[i * D + d] = dG_dU * (s_grad[d] - s_i[d] * s_dot_grad);
    }

    // Write outputs
    scalar_fields[i] = vec4f(U_accum, G_val, R_accum, E_val);
    grad_e_out[i] = grad_E;
}
