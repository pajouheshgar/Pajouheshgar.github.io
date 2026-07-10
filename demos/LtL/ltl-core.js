'use strict';
// Shared simulation core for the Larger-than-Life demo (index.html) and its
// correctness test (test.html).
//
// State texture: r = alive (0/1), g = fading display trail.
// Neighbor counting uses a summed-area table (SAT) of the alive channel,
// rebuilt every step with log2(N) Hillis-Steele scan passes per axis; the
// periodic (2R+1)^2 box sum is then 4-16 texel fetches per cell. The count
// includes the cell itself (Evans' convention).

// Periodic box sum via up to 4 SAT rectangles. Kept in fragment code (not
// Inc): the helpers reference the `sat` sampler, which is only auto-declared
// in stages that use it.
//
// All counting arithmetic is 32-bit integer. The SAT texels are float only
// because SwissGL fragment shaders can write nothing else (FOut is hardcoded
// vec4; integer textures would need ivec4 outputs + isampler access) — but
// the values stored are integer residues < 2048, which float32 represents
// exactly, so the float textures are just lossless containers.
//
// Residues: prefix sums are kept modulo (2048, 2047) in the two channels.
// Raw sums on large grids would exceed float32's exact-integer range (an
// 8192^2 grid reaches 67M > 2^24), while residues stay small on any grid.
// The count is recovered by CRT — since 2048 = 1 (mod 2047), count =
// ((y - x) mod 2047)*2048 + x — exact up to 2048*2047 ~ 4.2M, covering the
// (2R+1)^2 = 1025^2 maximum at R = 512.
//
// GLSL ES quirk: % is undefined for negative operands, so a multiple of the
// modulus is added before every reduction of a possibly-negative value.
const BOXSUM_GLSL = `
    const ivec2 SATM = ivec2(2048, 2047);
    ivec2 satAt(int x, int y) {
        return (x < 0 || y < 0) ? ivec2(0) : ivec2(sat(ivec2(x, y)).xy);
    }
    ivec2 rectSum(int x0, int x1, int y0, int y1) {   // inclusive, in-bounds
        return (satAt(x1, y1) - satAt(x0 - 1, y1)
              - satAt(x1, y0 - 1) + satAt(x0 - 1, y0 - 1) + 2*SATM) % SATM;
    }
    int boxSum(ivec2 c, int R, int n) {
        // wrap [c-R .. c+R] into 1-2 in-bounds segments per axis;
        // the b segment is empty (xb0 > xb1) when there is no wrap
        int x0 = c.x - R, x1 = c.x + R, y0 = c.y - R, y1 = c.y + R;
        int xa0 = max(x0, 0), xa1 = min(x1, n - 1), xb0 = 0, xb1 = -1;
        if (x0 < 0) { xb0 = x0 + n; xb1 = n - 1; } else if (x1 >= n) { xb0 = 0; xb1 = x1 - n; }
        int ya0 = max(y0, 0), ya1 = min(y1, n - 1), yb0 = 0, yb1 = -1;
        if (y0 < 0) { yb0 = y0 + n; yb1 = n - 1; } else if (y1 >= n) { yb0 = 0; yb1 = y1 - n; }
        ivec2 s = rectSum(xa0, xa1, ya0, ya1);    // each term is in [0, SATM)
        if (xb1 >= xb0) s += rectSum(xb0, xb1, ya0, ya1);
        if (yb1 >= yb0) s += rectSum(xa0, xa1, yb0, yb1);
        if (xb1 >= xb0 && yb1 >= yb0) s += rectSum(xb0, xb1, yb0, yb1);
        s %= SATM;                                // residues of the count
        return ((s.y - s.x + 2047) % 2047) * 2048 + s.x;
    }`;

function makeLtL(glsl, N) {
    const state = glsl({}, {size: [N, N], format: 'rgba16f', story: 2, tag: 'state' + N});
    // rg32f: prefix sums modulo (2048, 2047) — see BOXSUM_GLSL
    const sat = glsl({}, {size: [N, N], format: 'rg32f', story: 2, tag: 'sat' + N});

    // freq == 0 (or omitted): uniform white noise. freq > 0: periodic Perlin
    // noise with `freq` lattice cells across the grid (lower = bigger blobs),
    // thresholded so ~density of the cells are alive.
    function randomize(seed, density, freq) {
        glsl({seed: seed == null ? Math.floor(Math.random() * 1e6) : seed,
              density: density == null ? 0.5 : density, freq: freq || 0, FP: `
            vec2 grad(ivec2 c) {
                // wrap the lattice so the noise tiles with the periodic grid
                ivec2 cw = ivec2(mod(vec2(c), vec2(freq)));
                float a = hash(ivec3(cw, int(seed))).x * TAU;
                return vec2(cos(a), sin(a));
            }
            float perlin(vec2 p) {
                ivec2 i0 = ivec2(floor(p));
                vec2 f = fract(p), u = f*f*(3.0 - 2.0*f);
                float d00 = dot(grad(i0),              f);
                float d10 = dot(grad(i0 + ivec2(1,0)), f - vec2(1,0));
                float d01 = dot(grad(i0 + ivec2(0,1)), f - vec2(0,1));
                float d11 = dot(grad(i0 + ivec2(1,1)), f - vec2(1,1));
                return mix(mix(d00, d10, u.x), mix(d01, d11, u.x), u.y);
            }
            void fragment() {
                float v = freq > 0.5
                    ? clamp(perlin(UV * freq) / 0.7 * 0.5 + 0.5, 0.0, 1.0)
                    : hash(ivec3(I, int(seed))).x;
                FOut = vec4(step(1.0 - density, v), 0.0, 0.0, 1.0);
            }`}, state);
    }

    function clear() {
        glsl({FP: 'vec4(0)'}, state);
    }

    function buildSAT() {
        // load the alive channel, then inclusive prefix sums (Hillis-Steele),
        // horizontal then vertical, in residues mod (2048, 2047)
        glsl({cells: state[0], FP: 'vec2(cells(I).r), 0.0, 1.0'}, sat);
        for (let d = 1; d < N; d <<= 1)
            glsl({dist: d, FP: `
                ivec2 v = ivec2(Src(I).xy);
                int d = int(dist);
                if (I.x >= d) v = (v + ivec2(Src(ivec2(I.x - d, I.y)).xy))
                                  % ivec2(2048, 2047);
                FOut = vec4(v, 0.0, 1.0);`}, sat);
        for (let d = 1; d < N; d <<= 1)
            glsl({dist: d, FP: `
                ivec2 v = ivec2(Src(I).xy);
                int d = int(dist);
                if (I.y >= d) v = (v + ivec2(Src(ivec2(I.x, I.y - d)).xy))
                                  % ivec2(2048, 2047);
                FOut = vec4(v, 0.0, 1.0);`}, sat);
    }

    // rule = {R, b1, b2, s1, s2}
    function step(rule) {
        buildSAT();
        glsl({sat: sat[0], radius: rule.R,
              b1: rule.b1, b2: rule.b2, s1: rule.s1, s2: rule.s2,
              FP: BOXSUM_GLSL + `
            void fragment() {
                int cnt = boxSum(I, int(radius), ViewSize.x);
                bool alive = Src(I).r > 0.5;
                bool next = alive ? (cnt >= int(s1) && cnt <= int(s2))
                                  : (cnt >= int(b1) && cnt <= int(b2));
                float trail = next ? 1.0 : max(Src(I).g * 0.96 - 0.002, 0.0);
                FOut = vec4(next ? 1.0 : 0.0, trail, 0.0, 1.0);
            }`}, state);
    }

    // copy of the alive channel as a fresh Float32Array; box = [x, y, w, h]
    // texels (readSync reuses the texture's CPU buffer, hence the .slice())
    function readAlive(box) {
        return glsl({cells: state[0], FP: 'cells(I).r, 0.0, 0.0, 1.0'},
            {size: [N, N], format: 'r32f', tag: 'readback' + N})
            .readSync(...(box || [])).slice();
    }

    return {N, state, sat, randomize, clear, buildSAT, step, readAlive};
}
