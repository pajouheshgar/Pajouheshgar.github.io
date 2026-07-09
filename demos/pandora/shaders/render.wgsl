// Instanced point sprite rendering.
// Each particle = 1 instance = 6 vertices (2 triangles forming a quad).
// Color visualizes the first 3 channels of the state vector s (mapped from
// [-1,1] to [0,1]); the species type selects the sprite shape:
// type 0 = circle, type 1 = triangle, type 2 = square, type 3 = pentagon, ...

const PI: f32 = 3.14159265359;
// Global sprite scale: slider value 3.0 renders at what used to be 0.5
const POINT_SIZE_SCALE: f32 = 1.0 / 6.0;

struct RenderParams {
    pan: vec2f,
    zoom: f32,
    aspect: f32,
    point_size: f32,
    box_size: f32,
    state_dim: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> types: array<u32>;
@group(0) @binding(2) var<uniform> render_params: RenderParams;
@group(0) @binding(3) var<storage, read> states: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
    @location(2) @interpolate(flat) sides: u32,
};

struct BoundsVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

// Quad vertex offsets: two triangles
const QUAD_OFFSETS = array<vec2f, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
);

fn world_to_ndc(world_pos: vec2f) -> vec2f {
    let centered = (world_pos - render_params.pan) * render_params.zoom * 2.0 / render_params.box_size;

    var ndc = centered;
    if (render_params.aspect > 1.0) {
        ndc.x /= render_params.aspect;
    } else {
        ndc.y *= render_params.aspect;
    }
    return ndc;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = positions[instance_idx];
    let type_id = types[instance_idx];

    let ndc = world_to_ndc(world_pos);

    // Quad corner offset in clip space, aspect-corrected so sprites stay regular
    let corner = QUAD_OFFSETS[vertex_idx];
    var pixel_offset = corner * render_params.point_size * POINT_SIZE_SCALE * render_params.zoom / render_params.box_size;
    if (render_params.aspect > 1.0) {
        pixel_offset.x /= render_params.aspect;
    } else {
        pixel_offset.y *= render_params.aspect;
    }

    out.position = vec4f(ndc + pixel_offset, 0.0, 1.0);
    out.uv = corner;

    // Color from the first 3 state channels: [-1,1] -> [0,1]
    let D = u32(render_params.state_dim);
    let base = instance_idx * D;
    let s = vec3f(states[base], states[base + 1u], states[base + 2u]);
    out.color = clamp(s * 0.5 + vec3f(0.5), vec3f(0.0), vec3f(1.0));

    // Shape: type 0 -> circle (0 = circle sentinel), type k>=1 -> (k+2)-gon
    out.sides = select(type_id + 2u, 0u, type_id == 0u);

    return out;
}

@vertex
fn bounds_vs(@builtin(vertex_index) vertex_idx: u32) -> BoundsVertexOutput {
    var out: BoundsVertexOutput;

    let h = render_params.box_size * 0.5;
    var world_pos = vec2f(-h, -h);
    switch (vertex_idx) {
        case 0u: { world_pos = vec2f(-h, -h); }
        case 1u: { world_pos = vec2f( h, -h); }
        case 2u: { world_pos = vec2f( h,  h); }
        case 3u: { world_pos = vec2f(-h,  h); }
        default: { world_pos = vec2f(-h, -h); }
    }

    out.position = vec4f(world_to_ndc(world_pos), 0.0, 1.0);
    out.color = vec3f(0.65, 0.75, 0.95);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let r = length(in.uv);

    // Normalized distance: 1.0 at the shape boundary.
    var d = r;
    if (in.sides != 0u) {
        // Regular n-gon with circumradius 1, one vertex pointing up.
        let n = f32(in.sides);
        let seg = 2.0 * PI / n;
        var ang = atan2(in.uv.y, in.uv.x) - 0.5 * PI;
        ang = ang - seg * floor(ang / seg);
        d = r * cos(ang - 0.5 * seg) / cos(0.5 * seg);
    }

    // Fully opaque with a ~1px anti-aliased boundary
    let aa = max(fwidth(d), 1e-4);
    let alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    if (alpha <= 0.001) { discard; }
    return vec4f(in.color * alpha, alpha);
}

@fragment
fn bounds_fs(in: BoundsVertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 0.85);
}
