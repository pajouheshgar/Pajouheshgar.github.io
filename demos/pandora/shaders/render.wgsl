// Instanced point sprite rendering.
// Each particle = 1 instance = 6 vertices (2 triangles forming a quad).

struct RenderParams {
    pan: vec2f,
    zoom: f32,
    aspect: f32,
    point_size: f32,
    box_size: f32,
    _pad0: f32,
    _pad1: f32,
};

struct PaletteEntry {
    color: vec4f,
};

@group(0) @binding(0) var<storage, read> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read> types: array<u32>;
@group(0) @binding(2) var<uniform> render_params: RenderParams;
@group(0) @binding(3) var<storage, read> palette: array<PaletteEntry>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec3f,
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

    // Quad corner offset in clip space
    let corner = QUAD_OFFSETS[vertex_idx];
    let pixel_offset = corner * render_params.point_size * render_params.zoom / render_params.box_size;

    out.position = vec4f(ndc + pixel_offset, 0.0, 1.0);
    out.uv = corner;
    out.color = palette[type_id % 10u].color.rgb;

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
    let r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) { discard; }

    // Gaussian falloff
    let alpha = exp(-6.0 * r2);
    return vec4f(in.color * alpha, alpha);
}

@fragment
fn bounds_fs(in: BoundsVertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 0.85);
}
