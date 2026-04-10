/**
 * Spatial hash grid configuration and buffer management.
 */

/**
 * Pack SimParams into a uniform buffer.
 * Must match the WGSL SimParams struct layout exactly.
 *
 * struct SimParams {
 *   N: u32,             // offset 0
 *   cell_count: u32,    // offset 4
 *   num_species: u32,   // offset 8
 *   grid_size: u32,     // offset 12
 *   log2_grid_size: u32, // offset 16
 *   cell_radius: i32,   // offset 20
 *   box_size: f32,      // offset 24
 *   half_box: f32,      // offset 28
 *   eps: f32,           // offset 32
 *   inv_eps: f32,       // offset 36
 *   R_max_sq: f32,      // offset 40
 *   dt: f32,            // offset 44
 * };
 * Total: 48 bytes (12 x 4)
 */
export function packSimParams(config) {
    const buf = new ArrayBuffer(48);
    const u32 = new Uint32Array(buf);
    const i32 = new Int32Array(buf);
    const f32 = new Float32Array(buf);

    u32[0] = config.N;
    u32[1] = config.cellCount;
    u32[2] = config.numSpecies;
    u32[3] = config.gridSize;
    u32[4] = config.log2GridSize;
    i32[5] = config.cellRadius;
    f32[6] = config.boxSize;
    f32[7] = config.halfBox;
    f32[8] = config.eps;
    f32[9] = config.invEps;
    f32[10] = config.RMaxSq;
    f32[11] = config.dt;

    return new Uint8Array(buf);
}

/**
 * Compute grid configuration from simulation parameters.
 */
export function computeGridConfig(N, numSpecies, gridSize, boxSize, dt, rMax) {
    const eps = boxSize / gridSize;
    const cellRadius = Math.ceil(rMax / eps);
    const cellCount = gridSize * gridSize;
    const log2GridSize = Math.log2(gridSize);

    if (!Number.isInteger(log2GridSize)) {
        throw new Error(`gridSize must be power of 2, got ${gridSize}`);
    }

    // Check that search diameter doesn't exceed grid size
    const searchDiam = 2 * cellRadius + 1;
    if (searchDiam > gridSize) {
        throw new Error(
            `Interaction range too large: search diameter ${searchDiam} > grid_size ${gridSize}. ` +
            `Increase grid_size or reduce interaction range.`
        );
    }

    return {
        N,
        cellCount,
        numSpecies,
        gridSize,
        log2GridSize,
        cellRadius,
        boxSize,
        halfBox: boxSize * 0.5,
        eps,
        invEps: 1.0 / eps,
        RMaxSq: rMax * rMax,
        dt,
    };
}
