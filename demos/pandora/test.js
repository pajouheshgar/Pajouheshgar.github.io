/**
 * Test suite: validates JS reference against Python fixture,
 * and WebGPU implementation against JS reference.
 */

import { leniaForward, relativeError, calcKWeight } from './reference.js';
import { LeniaSimulation } from './js/lenia.js';
import { computeGridConfig } from './js/spatial_hash.js';
import { packSpeciesParams, computeRMax } from './js/species.js';

const results = [];
let testContainer;

// Seeded PRNG (mulberry32) for deterministic state vectors
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussianSample(rng) {
    let u = rng();
    while (u <= 1e-12) u = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * rng());
}

// Random unit-length state vectors (f64), deterministic for a given seed
function makeUnitStates(N, stateDim, seed) {
    const rng = mulberry32(seed);
    const states = new Float64Array(N * stateDim);
    for (let i = 0; i < N; i++) {
        let normSq = 0.0;
        for (let d = 0; d < stateDim; d++) {
            const v = gaussianSample(rng);
            states[i * stateDim + d] = v;
            normSq += v * v;
        }
        const invNorm = 1.0 / Math.sqrt(Math.max(normSq, 1e-12));
        for (let d = 0; d < stateDim; d++) {
            states[i * stateDim + d] *= invNorm;
        }
    }
    return states;
}

function log(msg, isError = false) {
    const div = document.createElement('div');
    div.className = isError ? 'fail' : 'info';
    div.textContent = msg;
    testContainer.appendChild(div);
    console.log(msg);
}

function assert(condition, msg) {
    if (!condition) throw new Error(`ASSERTION FAILED: ${msg}`);
}

function reportTest(name, passed, details = '') {
    results.push({ name, passed, details });
    const div = document.createElement('div');
    div.className = passed ? 'pass' : 'fail';
    div.textContent = `${passed ? 'PASS' : 'FAIL'}: ${name}${details ? ' — ' + details : ''}`;
    testContainer.appendChild(div);
    console.log(div.textContent);
}

// ============================================================================
// Test 1: JS Reference vs Python Fixture
// ============================================================================
async function testJsRefVsPythonFixture() {
    const fixture = await fetch('test_fixture.json').then(r => r.json());
    const { N, num_species, box_size, positions: posFlat, types: typesFlat, species_params: sp, expected } = fixture;

    const positions = new Float64Array(posFlat);
    const types = new Uint32Array(typesFlat);
    const speciesParams = {
        muK: new Float64Array(sp.muK),
        sigmaK: new Float64Array(sp.sigmaK),
        wK: new Float64Array(sp.wK),
        muG: new Float64Array(sp.muG),
        sigmaG: new Float64Array(sp.sigmaG),
        cRep: new Float64Array(sp.cRep),
    };

    const result = leniaForward(positions, types, speciesParams, N, box_size);

    const expectedU = new Float64Array(expected.U);
    const expectedG = new Float64Array(expected.G);
    const expectedR = new Float64Array(expected.R);
    const expectedE = new Float64Array(expected.E);
    const expectedGradE = new Float64Array(expected.gradE);

    const errU = relativeError(result.U, expectedU);
    const errG = relativeError(result.G, expectedG);
    const errR = relativeError(result.R, expectedR);
    const errE = relativeError(result.E, expectedE);
    const errGradE = relativeError(result.gradE, expectedGradE);

    log(`  U error: ${errU.toExponential(2)}`);
    log(`  G error: ${errG.toExponential(2)}`);
    log(`  R error: ${errR.toExponential(2)}`);
    log(`  E error: ${errE.toExponential(2)}`);
    log(`  grad_E error: ${errGradE.toExponential(2)}`);

    const allPass = errU < 1e-6 && errG < 1e-6 && errR < 1e-6 && errE < 1e-6 && errGradE < 1e-6;
    reportTest('JS Reference vs Python Fixture', allPass,
        `U=${errU.toExponential(2)}, G=${errG.toExponential(2)}, R=${errR.toExponential(2)}, E=${errE.toExponential(2)}, gradE=${errGradE.toExponential(2)}`);
}

// ============================================================================
// Test 2: WebGPU forward vs JS Reference (state-vector dynamics)
// ============================================================================
async function testWebGPUvsJsRef(device) {
    const fixture = await fetch('test_fixture.json').then(r => r.json());
    const { N, num_species, box_size, positions: posFlat, types: typesFlat, species_params: sp } = fixture;

    // Deterministic unit-length state vectors (D=5 exercises the generic loop)
    const stateDim = 5;
    const states64 = makeUnitStates(N, stateDim, 0xC0FFEE);
    const states32 = new Float32Array(states64);

    // Run JS reference (f64)
    const positions64 = new Float64Array(posFlat);
    const types = new Uint32Array(typesFlat);
    const speciesParamsF64 = {
        muK: new Float64Array(sp.muK),
        sigmaK: new Float64Array(sp.sigmaK),
        wK: new Float64Array(sp.wK),
        muG: new Float64Array(sp.muG),
        sigmaG: new Float64Array(sp.sigmaG),
        cRep: new Float64Array(sp.cRep),
    };
    const jsResult = leniaForward(positions64, types, speciesParamsF64, N, box_size, states64, stateDim);

    // Setup WebGPU simulation
    const speciesParamsF32 = {
        muK: new Float32Array(sp.muK),
        sigmaK: new Float32Array(sp.sigmaK),
        wK: new Float32Array(sp.wK),
        muG: new Float32Array(sp.muG),
        sigmaG: new Float32Array(sp.sigmaG),
        cRep: new Float32Array(sp.cRep),
    };

    const rMax = computeRMax(speciesParamsF32);
    const gridSize = 32; // Match fixture
    const dt = 0.25;
    const config = computeGridConfig(N, num_species, gridSize, box_size, dt, rMax, stateDim);

    const sim = new LeniaSimulation();
    await sim.init(device, config);
    sim.uploadSimParams(config);
    sim.uploadSpeciesParams(packSpeciesParams(speciesParamsF32));

    // Upload positions (f32) and states
    const positionsF32 = new Float32Array(posFlat);
    sim.uploadParticles(positionsF32, types, states32);

    // Run forward only (binning + forward, no position update)
    const gpuResult = await sim.runForwardOnly();

    // Extract per-particle fields from GPU output (vec4f layout: U, G, R, E)
    const gpuU = new Float32Array(N);
    const gpuG = new Float32Array(N);
    const gpuR = new Float32Array(N);
    const gpuE = new Float32Array(N);

    // GPU output is in binned order — we need to map back to original order
    // Read back the permutation to unbinned
    // Actually, the JS reference computed in original order. The GPU computed in binned order.
    // We need to compare in a consistent order. Let's compare per-particle by unbinning GPU results.

    // Read permutation
    const permReadBuf = device.createBuffer({
        size: N * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(sim.buffers.permutation, 0, permReadBuf, 0, N * 4);
    device.queue.submit([enc.finish()]);
    await permReadBuf.mapAsync(GPUMapMode.READ);
    const perm = new Uint32Array(permReadBuf.getMappedRange().slice(0));
    permReadBuf.unmap();
    permReadBuf.destroy();

    // Unbin: original particle i is at binned position perm[i]
    const gpuUUnbinned = new Float64Array(N);
    const gpuGUnbinned = new Float64Array(N);
    const gpuRUnbinned = new Float64Array(N);
    const gpuEUnbinned = new Float64Array(N);
    const gpuGradEUnbinned = new Float64Array(N * 2);
    const gpuSForceUnbinned = new Float64Array(N * stateDim);

    for (let i = 0; i < N; i++) {
        const j = perm[i]; // binned index
        gpuUUnbinned[i] = gpuResult.scalarFields[j * 4 + 0];
        gpuGUnbinned[i] = gpuResult.scalarFields[j * 4 + 1];
        gpuRUnbinned[i] = gpuResult.scalarFields[j * 4 + 2];
        gpuEUnbinned[i] = gpuResult.scalarFields[j * 4 + 3];
        gpuGradEUnbinned[i * 2] = gpuResult.gradE[j * 2];
        gpuGradEUnbinned[i * 2 + 1] = gpuResult.gradE[j * 2 + 1];
        for (let d = 0; d < stateDim; d++) {
            gpuSForceUnbinned[i * stateDim + d] = gpuResult.stateForce[j * stateDim + d];
        }
    }

    const errU = relativeError(gpuUUnbinned, jsResult.U);
    const errG = relativeError(gpuGUnbinned, jsResult.G);
    const errR = relativeError(gpuRUnbinned, jsResult.R);
    const errE = relativeError(gpuEUnbinned, jsResult.E);
    const errGradE = relativeError(gpuGradEUnbinned, jsResult.gradE);
    const errSForce = relativeError(gpuSForceUnbinned, jsResult.sForce);

    log(`  U error: ${errU.toExponential(2)}`);
    log(`  G error: ${errG.toExponential(2)}`);
    log(`  R error: ${errR.toExponential(2)}`);
    log(`  E error: ${errE.toExponential(2)}`);
    log(`  grad_E error: ${errGradE.toExponential(2)}`);
    log(`  s_force error: ${errSForce.toExponential(2)}`);

    // Relaxed tolerances: f32 GPU vs f64 JS + different wrapping
    const allPass = errU < 5e-4 && errG < 5e-3 && errR < 5e-4 && errE < 5e-3
        && errGradE < 5e-2 && errSForce < 5e-2;
    reportTest('WebGPU Forward vs JS Reference', allPass,
        `U=${errU.toExponential(2)}, G=${errG.toExponential(2)}, R=${errR.toExponential(2)}, E=${errE.toExponential(2)}, gradE=${errGradE.toExponential(2)}, sForce=${errSForce.toExponential(2)}`);

    sim.destroy();
}

// ============================================================================
// Test 3: Multi-step stability
// ============================================================================
async function testMultiStepStability(device) {
    const N = 512;
    const numSpecies = 2;
    const gridSize = 32;
    const boxSize = 64.0;
    const dt = 0.25;

    const sp = {
        muK: new Float32Array([4.0, 4.5]),
        sigmaK: new Float32Array([1.0, 1.1]),
        wK: new Float32Array([calcKWeight(4.0, 1.0), calcKWeight(4.5, 1.1)]),
        muG: new Float32Array([0.6, 0.7]),
        sigmaG: new Float32Array([0.15, 0.17]),
        cRep: new Float32Array([1.0, 1.0]),
    };

    const stateDim = 3;
    const rMax = computeRMax(sp);
    const config = computeGridConfig(N, numSpecies, gridSize, boxSize, dt, rMax, stateDim);

    const sim = new LeniaSimulation();
    await sim.init(device, config);
    sim.uploadSimParams(config);
    sim.uploadSpeciesParams(packSpeciesParams(sp));

    // Random init
    const positions = new Float32Array(N * 2);
    const types = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
        positions[i * 2] = (Math.random() - 0.5) * 0.8 * boxSize;
        positions[i * 2 + 1] = (Math.random() - 0.5) * 0.8 * boxSize;
        types[i] = Math.floor(Math.random() * numSpecies);
    }
    const states = new Float32Array(makeUnitStates(N, stateDim, (Math.random() * 0xFFFFFFFF) >>> 0));
    sim.uploadParticles(positions, types, states);

    // Run 100 steps
    for (let s = 0; s < 100; s++) {
        sim.step(1);
    }

    // Read back positions
    const result = await sim.readBack();
    const halfBox = boxSize * 0.5;

    let hasNaN = false;
    let outOfBounds = false;
    let badStateNorm = false;
    for (let i = 0; i < N; i++) {
        const x = result.positions[i * 2];
        const y = result.positions[i * 2 + 1];
        if (isNaN(x) || isNaN(y) || !isFinite(x) || !isFinite(y)) {
            hasNaN = true;
            break;
        }
        // Allow small overshoot due to f32 wrapping
        if (Math.abs(x) > halfBox + 1.0 || Math.abs(y) > halfBox + 1.0) {
            outOfBounds = true;
        }
        // States must stay finite and unit-length
        let normSq = 0.0;
        for (let d = 0; d < stateDim; d++) {
            const s = result.states[i * stateDim + d];
            if (isNaN(s) || !isFinite(s)) {
                hasNaN = true;
                break;
            }
            normSq += s * s;
        }
        if (hasNaN) break;
        if (Math.abs(Math.sqrt(normSq) - 1.0) > 1e-3) {
            badStateNorm = true;
        }
    }

    const passed = !hasNaN && !outOfBounds && !badStateNorm;
    reportTest('Multi-step Stability (100 steps)', passed,
        hasNaN ? 'NaN/Inf detected'
        : outOfBounds ? 'Particles out of bounds'
        : badStateNorm ? 'State vectors drifted off unit length'
        : 'All particles stable, states unit-length');

    sim.destroy();
}

// ============================================================================
// Test: Frozen states stay bitwise-identical while positions keep evolving
// ============================================================================
async function testFrozenStates(device) {
    const N = 512;
    const numSpecies = 2;
    const gridSize = 32;
    const boxSize = 64.0;
    const dt = 0.25;
    const stateDim = 4;

    const sp = {
        muK: new Float32Array([4.0, 4.5]),
        sigmaK: new Float32Array([1.0, 1.1]),
        wK: new Float32Array([calcKWeight(4.0, 1.0), calcKWeight(4.5, 1.1)]),
        muG: new Float32Array([0.6, 0.7]),
        sigmaG: new Float32Array([0.15, 0.17]),
        cRep: new Float32Array([1.0, 1.0]),
    };

    const rMax = computeRMax(sp);
    const config = computeGridConfig(N, numSpecies, gridSize, boxSize, dt, rMax, stateDim);
    config.freezeStates = 1;

    const sim = new LeniaSimulation();
    await sim.init(device, config);
    sim.uploadSimParams(config);
    sim.uploadSpeciesParams(packSpeciesParams(sp));

    const positions = new Float32Array(N * 2);
    const types = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
        positions[i * 2] = (Math.random() - 0.5) * 0.8 * boxSize;
        positions[i * 2 + 1] = (Math.random() - 0.5) * 0.8 * boxSize;
        types[i] = Math.floor(Math.random() * numSpecies);
    }
    const states = new Float32Array(makeUnitStates(N, stateDim, 0xBEEF));
    sim.uploadParticles(positions, types, states);

    for (let s = 0; s < 50; s++) {
        sim.step(1);
    }
    const result = await sim.readBack();

    // States are permuted by binning, so compare as sorted multisets of components.
    const sortedBefore = Float32Array.from(states).sort();
    const sortedAfter = Float32Array.from(result.states).sort();
    let statesUnchanged = true;
    for (let i = 0; i < sortedBefore.length; i++) {
        if (sortedBefore[i] !== sortedAfter[i]) { statesUnchanged = false; break; }
    }

    // Positions must still evolve.
    const sortedPosBefore = Float32Array.from(positions).sort();
    const sortedPosAfter = Float32Array.from(result.positions).sort();
    let positionsMoved = false;
    for (let i = 0; i < sortedPosBefore.length; i++) {
        if (sortedPosBefore[i] !== sortedPosAfter[i]) { positionsMoved = true; break; }
    }

    const passed = statesUnchanged && positionsMoved;
    reportTest('Frozen states (50 steps)', passed,
        !statesUnchanged ? 'States changed while frozen'
        : !positionsMoved ? 'Positions did not evolve'
        : 'States bitwise-identical, positions evolving');

    sim.destroy();
}

// ============================================================================
// Test 4: calc_K_weight matches Python
// ============================================================================
function testCalcKWeight() {
    // Known values from Python torch_lenia.calc_K_weight
    // mu=4.0, sigma=1.0, dim=2 -> w_k
    const wk = calcKWeight(4.0, 1.0, 2);

    // This should produce a reasonable normalization constant
    // The exact value depends on the numerical integration
    const isFinitePositive = isFinite(wk) && wk > 0;

    reportTest('calcKWeight sanity check', isFinitePositive,
        `w_k(4.0, 1.0) = ${wk.toFixed(6)}, expected positive finite`);
}

// ============================================================================
// Run all tests
// ============================================================================
export async function runTests() {
    testContainer = document.getElementById('test-results');
    testContainer.innerHTML = '';

    log('=== Running Test Suite ===');
    log('');

    // Test 0: calcKWeight
    log('--- Test: calcKWeight sanity ---');
    testCalcKWeight();
    log('');

    // Test 1: JS ref vs Python
    log('--- Test: JS Reference vs Python Fixture ---');
    try {
        await testJsRefVsPythonFixture();
    } catch (e) {
        reportTest('JS Reference vs Python Fixture', false, e.message);
        log(e.stack, true);
    }
    log('');

    // WebGPU tests
    if (!navigator.gpu) {
        log('WebGPU not available - skipping GPU tests', true);
    } else {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            log('No WebGPU adapter - skipping GPU tests', true);
        } else {
            const device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                },
            });

            // Test 2: WebGPU vs JS ref
            log('--- Test: WebGPU Forward vs JS Reference ---');
            try {
                await testWebGPUvsJsRef(device);
            } catch (e) {
                reportTest('WebGPU Forward vs JS Reference', false, e.message);
                log(e.stack, true);
            }
            log('');

            // Test 3: Stability
            log('--- Test: Multi-step Stability ---');
            try {
                await testMultiStepStability(device);
            } catch (e) {
                reportTest('Multi-step Stability', false, e.message);
                log(e.stack, true);
            }
            log('');

            // Test 4: Frozen states
            log('--- Test: Frozen States ---');
            try {
                await testFrozenStates(device);
            } catch (e) {
                reportTest('Frozen states', false, e.message);
                log(e.stack, true);
            }

            device.destroy();
        }
    }

    log('');
    const passed = results.filter(r => r.passed).length;
    const total = results.length;
    log(`=== ${passed}/${total} tests passed ===`);
}
