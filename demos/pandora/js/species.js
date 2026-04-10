/**
 * Species parameter generation and normalization.
 */

const KERNEL_SIGMA_CUTOFF = 3.0;

/**
 * Compute w_k normalization constant (matches torch_lenia.py calc_K_weight).
 */
export function calcKWeight(mu, sigma, dim = 2) {
    const rMin = Math.max(mu - sigma * 4, 0.0);
    const rMax = mu + sigma * 4;
    const steps = 200;
    const dr = (rMax - rMin) / steps;

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
 * Generate default species parameters (matches glfw_interop_sim.py make_species_params).
 * @returns {{ muK, sigmaK, wK, muG, sigmaG, cRep }} each Float32Array[numSpecies]
 */
export function makeSpeciesParams(numSpecies) {
    const muK = new Float32Array(numSpecies);
    const sigmaK = new Float32Array(numSpecies);
    const muG = new Float32Array(numSpecies);
    const sigmaG = new Float32Array(numSpecies);
    const cRep = new Float32Array(numSpecies);
    const wK = new Float32Array(numSpecies);

    for (let i = 0; i < numSpecies; i++) {
        muK[i] = i === 0 ? 0.0 : 4.0 + 0.5 * i;
        sigmaK[i] = i === 0 ? 3.0 : 1.0 + 0.1 * i;
        muG[i] = 0.6 + 0.10 * i;
        sigmaG[i] = 0.15 + 0.02 * i;
        cRep[i] = 1.0;
        wK[i] = calcKWeight(muK[i], sigmaK[i]);
    }

    return { muK, sigmaK, wK, muG, sigmaG, cRep };
}

/**
 * Generate random species parameters (matches glfw_interop_sim.py make_random_species_params).
 */
export function makeRandomSpeciesParams(numSpecies) {
    const muK = new Float32Array(numSpecies);
    const sigmaK = new Float32Array(numSpecies);
    const muG = new Float32Array(numSpecies);
    const sigmaG = new Float32Array(numSpecies);
    const cRep = new Float32Array(numSpecies);
    const wK = new Float32Array(numSpecies);

    for (let i = 0; i < numSpecies; i++) {
        muK[i] = i === 0 ? 0.0 : 0.25 + Math.random() * 7.75;
        sigmaK[i] = 0.25 + Math.random() * 3.75;
        muG[i] = 0.10 + Math.random() * 1.20;
        sigmaG[i] = 0.05 + Math.random() * 0.45;
        cRep[i] = 0.30 + Math.random() * 1.70;
        wK[i] = calcKWeight(muK[i], sigmaK[i]);
    }

    return { muK, sigmaK, wK, muG, sigmaG, cRep };
}

/**
 * Pack species params into a flat Float32Array for GPU upload.
 * Layout per species: [mu_k, sigma_k, w_k, mu_g, sigma_g, c_rep, r_min_k, r_max_k]
 * 8 floats per species (32 bytes, naturally aligned).
 */
export function packSpeciesParams(params) {
    const M = params.muK.length;
    const data = new Float32Array(M * 8);
    for (let i = 0; i < M; i++) {
        const off = i * 8;
        data[off + 0] = params.muK[i];
        data[off + 1] = params.sigmaK[i];
        data[off + 2] = params.wK[i];
        data[off + 3] = params.muG[i];
        data[off + 4] = params.sigmaG[i];
        data[off + 5] = params.cRep[i];
        data[off + 6] = Math.max(params.muK[i] - KERNEL_SIGMA_CUTOFF * params.sigmaK[i], 0.0);
        data[off + 7] = params.muK[i] + KERNEL_SIGMA_CUTOFF * params.sigmaK[i];
    }
    return data;
}

/**
 * Compute maximum interaction radius from species params.
 */
export function computeRMax(params) {
    let rMax = 0;
    for (let i = 0; i < params.muK.length; i++) {
        const r = params.muK[i] + KERNEL_SIGMA_CUTOFF * params.sigmaK[i];
        if (r > rMax) rMax = r;
    }
    return rMax;
}

/** Color palette matching glfw_interop_sim.py build_palette */
export const PALETTE = [
    [235 / 255, 64 / 255, 52 / 255],
    [64 / 255, 178 / 255, 255 / 255],
    [52 / 255, 222 / 255, 135 / 255],
    [255 / 255, 194 / 255, 41 / 255],
    [181 / 255, 92 / 255, 255 / 255],
    [255 / 255, 120 / 255, 45 / 255],
    [52 / 255, 221 / 255, 221 / 255],
    [255 / 255, 87 / 255, 163 / 255],
    [133 / 255, 219 / 255, 63 / 255],
    [255 / 255, 255 / 255, 255 / 255],
];
