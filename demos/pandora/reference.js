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
 * Brute-force lenia forward pass.
 *
 * Legacy mode (states omitted): matches torch_lenia.py lenia_forward exactly
 * (no self-term, no state weighting) — used to validate against the Python fixture.
 *
 * State mode (states + stateDim given): matches the state-vector dynamics in
 * forward.wgsl — U starts with a self-term, pairwise U contributions are
 * weighted by relu(dot(s_i, s_j)), and a tangential state force
 * dG/dU * (s_grad - s * <s, s_grad>) is returned as sForce.
 *
 * @param {Float64Array} positions - N*2 interleaved [x0,y0, x1,y1, ...]
 * @param {Uint32Array} types - N particle species indices
 * @param {Object} speciesParams - { muK, sigmaK, wK, muG, sigmaG, cRep } each Float64Array[M]
 * @param {number} N - particle count
 * @param {number} boxSize - periodic domain size
 * @param {Float64Array|null} states - N*stateDim unit-length state vectors (optional)
 * @param {number} stateDim - state vector dimension (required if states given)
 * @returns {{ U, G, R, E, gradE, sForce }} (sForce is null in legacy mode)
 */
export function leniaForward(positions, types, speciesParams, N, boxSize, states = null, stateDim = 0) {
    const { muK, sigmaK, wK, muG, sigmaG, cRep } = speciesParams;
    const halfBox = boxSize * 0.5;
    const useStates = states !== null && stateDim > 0;

    const U = new Float64Array(N);
    const R = new Float64Array(N);
    const gradU = new Float64Array(N * 2);
    const gradR = new Float64Array(N * 2);
    const sGrad = useStates ? new Float64Array(N * stateDim) : null;

    // Pairwise interactions
    for (let i = 0; i < N; i++) {
        const xi = positions[i * 2];
        const yi = positions[i * 2 + 1];
        const typeI = types[i];

        // Self-term (state weight = dot(s,s) = 1)
        if (useStates) {
            const t0 = muK[typeI] / sigmaK[typeI];
            U[i] += wK[typeI] * Math.exp(-t0 * t0);
        }

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
                let sw = 1.0;
                let swRaw = 1.0;
                if (useStates) {
                    swRaw = 0.0;
                    for (let d = 0; d < stateDim; d++) {
                        swRaw += states[i * stateDim + d] * states[j * stateDim + d];
                    }
                    sw = Math.max(swRaw, 0.0);
                }

                const t = (r - muKj) / sigmaKj;
                const pf = Math.exp(-t * t);
                const dpfDr = pf * (-2.0 * t / sigmaKj);

                U[i] += wKj * pf * sw;

                const factor = wKj * dpfDr * invR * sw;
                gradU[i * 2] -= factor * dx;
                gradU[i * 2 + 1] -= factor * dy;

                if (useStates && swRaw > 0.0) {
                    const kw = wKj * pf;
                    for (let d = 0; d < stateDim; d++) {
                        sGrad[i * stateDim + d] += kw * states[j * stateDim + d];
                    }
                }
            }

            // Repulsion contribution (not state-weighted)
            if (r < 1.0) {
                const omr = 1.0 - r;
                R[i] += 0.5 * cRepJ * omr * omr;

                const repFactor = cRepJ * omr * invR;
                gradR[i * 2] += repFactor * dx;
                gradR[i * 2 + 1] += repFactor * dy;
            }
        }
    }

    // Growth, energy, gradient, and state force
    const G = new Float64Array(N);
    const E = new Float64Array(N);
    const gradE = new Float64Array(N * 2);
    const sForce = useStates ? new Float64Array(N * stateDim) : null;

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

        if (useStates) {
            let sDotGrad = 0.0;
            for (let d = 0; d < stateDim; d++) {
                sDotGrad += states[i * stateDim + d] * sGrad[i * stateDim + d];
            }
            for (let d = 0; d < stateDim; d++) {
                sForce[i * stateDim + d] = dGdU *
                    (sGrad[i * stateDim + d] - states[i * stateDim + d] * sDotGrad);
            }
        }
    }

    return { U, G, R, E, gradE, sForce };
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
