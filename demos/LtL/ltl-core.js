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
const BOXSUM_GLSL = `
    float satAt(int x, int y) {
        return (x < 0 || y < 0) ? 0.0 : sat(ivec2(x, y)).x;
    }
    float rectSum(int x0, int x1, int y0, int y1) {   // inclusive, in-bounds
        return satAt(x1, y1) - satAt(x0 - 1, y1)
             - satAt(x1, y0 - 1) + satAt(x0 - 1, y0 - 1);
    }
    float boxSum(ivec2 c, int R, int n) {
        // wrap [c-R .. c+R] into 1-2 in-bounds segments per axis;
        // the b segment is empty (xb0 > xb1) when there is no wrap
        int x0 = c.x - R, x1 = c.x + R, y0 = c.y - R, y1 = c.y + R;
        int xa0 = max(x0, 0), xa1 = min(x1, n - 1), xb0 = 0, xb1 = -1;
        if (x0 < 0) { xb0 = x0 + n; xb1 = n - 1; } else if (x1 >= n) { xb0 = 0; xb1 = x1 - n; }
        int ya0 = max(y0, 0), ya1 = min(y1, n - 1), yb0 = 0, yb1 = -1;
        if (y0 < 0) { yb0 = y0 + n; yb1 = n - 1; } else if (y1 >= n) { yb0 = 0; yb1 = y1 - n; }
        float s = rectSum(xa0, xa1, ya0, ya1);
        if (xb1 >= xb0) s += rectSum(xb0, xb1, ya0, ya1);
        if (yb1 >= yb0) s += rectSum(xa0, xa1, yb0, yb1);
        if (xb1 >= xb0 && yb1 >= yb0) s += rectSum(xb0, xb1, yb0, yb1);
        return s;
    }`;

function makeLtL(glsl, N) {
    const state = glsl({}, {size: [N, N], format: 'rgba16f', story: 2, tag: 'state' + N});
    // r32f: counts up to N*N stay exact in float32
    const sat = glsl({}, {size: [N, N], format: 'r32f', story: 2, tag: 'sat' + N});

    function randomize(seed) {
        glsl({seed: seed == null ? Math.floor(Math.random() * 1e6) : seed, FP: `
            float r = hash(ivec3(I, int(seed))).x;
            FOut = vec4(step(0.5, r), 0.0, 0.0, 1.0);`}, state);
    }

    function clear() {
        glsl({FP: 'vec4(0)'}, state);
    }

    function buildSAT() {
        // load the alive channel, then inclusive prefix sums (Hillis-Steele),
        // horizontal then vertical
        glsl({cells: state[0], FP: 'cells(I).r, 0.0, 0.0, 1.0'}, sat);
        for (let d = 1; d < N; d <<= 1)
            glsl({dist: d, FP: `
                float v = Src(I).x;
                int d = int(dist);
                if (I.x >= d) v += Src(ivec2(I.x - d, I.y)).x;
                FOut = vec4(v, 0.0, 0.0, 1.0);`}, sat);
        for (let d = 1; d < N; d <<= 1)
            glsl({dist: d, FP: `
                float v = Src(I).x;
                int d = int(dist);
                if (I.y >= d) v += Src(ivec2(I.x, I.y - d)).x;
                FOut = vec4(v, 0.0, 0.0, 1.0);`}, sat);
    }

    // rule = {R, b1, b2, s1, s2}
    function step(rule) {
        buildSAT();
        glsl({sat: sat[0], radius: rule.R,
              b1: rule.b1, b2: rule.b2, s1: rule.s1, s2: rule.s2,
              FP: BOXSUM_GLSL + `
            void fragment() {
                float cnt = boxSum(I, int(radius), ViewSize.x);
                bool alive = Src(I).r > 0.5;
                bool next = alive ? (cnt >= s1 && cnt <= s2)
                                  : (cnt >= b1 && cnt <= b2);
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
