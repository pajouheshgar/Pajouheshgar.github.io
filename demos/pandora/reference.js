/**
 * Pure JavaScript O(N^2) brute-force reference implementation of multi-species particle lenia.
 * Matches torch_lenia.py exactly. Uses f64 for precision.
 */

export const KERNEL_SIGMA_CUTOFF = 3.0;

/**
 * Gaussian bump: exp(-((x - mu) / sigma)^2)
 */
export function peakF(x, mu, sigma) {
    const t = (x - mu) / sigma;
    return Math.exp(-t * t);
}

/**
 * Compute w_k normalization constant (matches torch_lenia.py calc_K_weight).
 * Numerically integrates peak_f(r, mu, sigma) * r^(dim-1) * surface_area.
 */
export function calcKWeight(mu, sigma, dim = 2) {
    const rMin = Math.max(mu - sigma * 4, 0.0);
    const rMax = mu + sigma * 4;
    const steps = 200;
    const dr = (rMax - rMin) / steps;

    // Trapezoidal integration
    let integral = 0.0;
    for (let i = 0; i <= steps; i++) {
        const r = rMin + i * dr;
        const t = (r - mu) / sigma;
        const val = Math.exp(-t * t) * Math.pow(r, dim - 1);
        const weight = (i === 0 || i === steps) ? 0.5 : 1.0;
        integral += val * weight * dr;
    }

    const surfaceArea = dim === 2 ? 2.0 * Math.PI : 4.0 * Math.PI;
    return 1.0 / (integral * surfaceArea);
}

/**
 * Brute-force lenia forward pass (matches torch_lenia.py lenia_forward).
 *
 * @param {Float64Array} positions - N*2 interleaved [x0,y0, x1,y1, ...]
 * @param {Uint32Array} types - N particle species indices
 * @param {Object} speciesParams - { muK, sigmaK, wK, muG, sigmaG, cRep } each Float64Array[M]
 * @param {number} N - particle count
 * @param {number} boxSize - periodic domain size
 * @returns {{ U: Float64Array, G: Float64Array, R: Float64Array, E: Float64Array, gradE: Float64Array }}
 */
export function leniaForward(positions, types, speciesParams, N, boxSize) {
    const { muK, sigmaK, wK, muG, sigmaG, cRep } = speciesParams;
    const halfBox = boxSize * 0.5;

    const U = new Float64Array(N);
    const R = new Float64Array(N);
    const gradU = new Float64Array(N * 2);
    const gradR = new Float64Array(N * 2);

    // Pairwise interactions
    for (let i = 0; i < N; i++) {
        const xi = positions[i * 2];
        const yi = positions[i * 2 + 1];

        for (let j = 0; j < N; j++) {
            if (j === i) continue;

            // Periodic displacement: dr = p_j - p_i
            let dx = positions[j * 2] - xi;
            let dy = positions[j * 2 + 1] - yi;

            // Wrap to [-halfBox, halfBox)
            // Match torch: remainder(dr + half_L, L) - half_L
            dx = ((dx + halfBox) % boxSize + boxSize) % boxSize - halfBox;
            dy = ((dy + halfBox) % boxSize + boxSize) % boxSize - halfBox;

            const rSq = dx * dx + dy * dy;
            const r = Math.sqrt(rSq);
            if (r < 1e-10) continue;

            const invR = 1.0 / r;
            const typeJ = types[j];

            const muKj = muK[typeJ];
            const sigmaKj = sigmaK[typeJ];
            const wKj = wK[typeJ];
            const cRepJ = cRep[typeJ];

            // Kernel support cutoff
            const rMinK = Math.max(muKj - KERNEL_SIGMA_CUTOFF * sigmaKj, 0.0);
            const rMaxK = muKj + KERNEL_SIGMA_CUTOFF * sigmaKj;

            // U-field contribution
            if (r >= rMinK && r <= rMaxK) {
                const t = (r - muKj) / sigmaKj;
                const pf = Math.exp(-t * t);
                const dpfDr = pf * (-2.0 * t / sigmaKj);

                U[i] += wKj * pf;

                const factor = wKj * dpfDr * invR;
                gradU[i * 2] -= factor * dx;
                gradU[i * 2 + 1] -= factor * dy;
            }

            // Repulsion contribution
            if (r < 1.0) {
                const omr = 1.0 - r;
                R[i] += 0.5 * cRepJ * omr * omr;

                const repFactor = cRepJ * omr * invR;
                gradR[i * 2] += repFactor * dx;
                gradR[i * 2 + 1] += repFactor * dy;
            }
        }
    }

    // Growth, energy, and gradient
    const G = new Float64Array(N);
    const E = new Float64Array(N);
    const gradE = new Float64Array(N * 2);

    for (let i = 0; i < N; i++) {
        const typeI = types[i];
        const muGi = muG[typeI];
        const sigmaGi = sigmaG[typeI];

        const tg = (U[i] - muGi) / sigmaGi;
        G[i] = Math.exp(-tg * tg);
        const dGdU = G[i] * (-2.0 * tg / sigmaGi);

        E[i] = R[i] - G[i];
        gradE[i * 2] = gradR[i * 2] - dGdU * gradU[i * 2];
        gradE[i * 2 + 1] = gradR[i * 2 + 1] - dGdU * gradU[i * 2 + 1];
    }

    return { U, G, R, E, gradE };
}

/**
 * Compute relative L2 error between two arrays.
 */
export function relativeError(a, b) {
    let diffSq = 0.0;
    let refSq = 0.0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        diffSq += d * d;
        refSq += b[i] * b[i];
    }
    const refNorm = Math.sqrt(refSq);
    if (refNorm < 1e-12) return Math.sqrt(diffSq);
    return Math.sqrt(diffSq) / refNorm;
}
