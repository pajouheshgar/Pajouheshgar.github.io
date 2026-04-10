/**
 * Core Lenia simulation: WebGPU setup, buffer management, compute dispatch.
 */

import { packSimParams, computeGridConfig } from './spatial_hash.js';
import { packSpeciesParams, computeRMax, makeSpeciesParams } from './species.js';

export class LeniaSimulation {
    constructor() {
        this.device = null;
        this.config = null;
        this.buffers = {};
        this.pipelines = {};
        this.bindGroups = {};
        this.paused = false;
        this.stepCount = 0;
    }

    async init(device, config) {
        this.device = device;
        this.config = config;

        // Load all shaders
        const [
            countCellsSrc,
            prefixSumSrc,
            computePermSrc,
            applyPermSrc,
            forwardSrc,
            updatePosSrc,
        ] = await Promise.all([
            fetch('shaders/count_cells.wgsl').then(r => r.text()),
            fetch('shaders/prefix_sum.wgsl').then(r => r.text()),
            fetch('shaders/compute_permutation.wgsl').then(r => r.text()),
            fetch('shaders/apply_permutation.wgsl').then(r => r.text()),
            fetch('shaders/forward.wgsl').then(r => r.text()),
            fetch('shaders/update_positions.wgsl').then(r => r.text()),
        ]);

        this._createBuffers();
        this._createPipelines(countCellsSrc, prefixSumSrc, computePermSrc,
                              applyPermSrc, forwardSrc, updatePosSrc);
        this._createBindGroups();
    }

    _createBuffers() {
        const { N, cellCount, numSpecies } = this.config;
        const d = this.device;

        // Positions (current, used as input to binning)
        this.buffers.positions = d.createBuffer({
            size: N * 8, // vec2f = 8 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Positions binned (output of scatter, input to forward)
        this.buffers.positionsBinned = d.createBuffer({
            size: N * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Types (current)
        this.buffers.types = d.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Types binned
        this.buffers.typesBinned = d.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Cell counts (atomic u32 per cell)
        this.buffers.cellCounts = d.createBuffer({
            size: cellCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Bin offsets (cell_count + 1 entries)
        this.buffers.binOffsets = d.createBuffer({
            size: (cellCount + 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Bin write indices (atomic u32 per cell)
        this.buffers.binWriteIndices = d.createBuffer({
            size: cellCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Permutation
        this.buffers.permutation = d.createBuffer({
            size: N * 4,
            // COPY_SRC is required for test readback in testWebGPUvsJsRef.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Species params
        this.buffers.speciesParams = d.createBuffer({
            size: numSpecies * 32, // 8 floats * 4 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // SimParams uniform
        this.buffers.simParams = d.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Scalar fields output (vec4f per particle: U, G, R, E)
        this.buffers.scalarFields = d.createBuffer({
            size: N * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Grad E output (vec2f per particle)
        this.buffers.gradE = d.createBuffer({
            size: N * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Readback staging buffers (for testing)
        this.buffers.scalarFieldsReadback = d.createBuffer({
            size: N * 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.buffers.gradEReadback = d.createBuffer({
            size: N * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.buffers.positionsReadback = d.createBuffer({
            size: N * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }

    _createPipelines(countCellsSrc, prefixSumSrc, computePermSrc,
                     applyPermSrc, forwardSrc, updatePosSrc) {
        const d = this.device;

        // --- Count cells pipeline ---
        const countCellsBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this.pipelines.countCells = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [countCellsBGL] }),
            compute: { module: d.createShaderModule({ code: countCellsSrc }), entryPoint: 'main' },
        });
        this.layouts = { countCells: countCellsBGL };

        // --- Prefix sum pipeline ---
        const prefixSumBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this.pipelines.prefixSum = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [prefixSumBGL] }),
            compute: { module: d.createShaderModule({ code: prefixSumSrc }), entryPoint: 'main' },
        });
        this.layouts.prefixSum = prefixSumBGL;

        // --- Compute permutation pipeline ---
        const computePermBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this.pipelines.computePerm = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [computePermBGL] }),
            compute: { module: d.createShaderModule({ code: computePermSrc }), entryPoint: 'main' },
        });
        this.layouts.computePerm = computePermBGL;

        // --- Apply permutation pipeline ---
        const applyPermBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this.pipelines.applyPerm = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [applyPermBGL] }),
            compute: { module: d.createShaderModule({ code: applyPermSrc }), entryPoint: 'main' },
        });
        this.layouts.applyPerm = applyPermBGL;

        // --- Forward pipeline ---
        const forwardBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        this.pipelines.forward = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [forwardBGL] }),
            compute: { module: d.createShaderModule({ code: forwardSrc }), entryPoint: 'main' },
        });
        this.layouts.forward = forwardBGL;

        // --- Update positions pipeline ---
        const updatePosBGL = d.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this.pipelines.updatePos = d.createComputePipeline({
            layout: d.createPipelineLayout({ bindGroupLayouts: [updatePosBGL] }),
            compute: { module: d.createShaderModule({ code: updatePosSrc }), entryPoint: 'main' },
        });
        this.layouts.updatePos = updatePosBGL;
    }

    _createBindGroups() {
        const d = this.device;
        const b = this.buffers;

        this.bindGroups.countCells = d.createBindGroup({
            layout: this.layouts.countCells,
            entries: [
                { binding: 0, resource: { buffer: b.positions } },
                { binding: 1, resource: { buffer: b.cellCounts } },
                { binding: 2, resource: { buffer: b.simParams } },
            ],
        });

        this.bindGroups.prefixSum = d.createBindGroup({
            layout: this.layouts.prefixSum,
            entries: [
                { binding: 0, resource: { buffer: b.cellCounts } },
                { binding: 1, resource: { buffer: b.binOffsets } },
                { binding: 2, resource: { buffer: b.simParams } },
            ],
        });

        this.bindGroups.computePerm = d.createBindGroup({
            layout: this.layouts.computePerm,
            entries: [
                { binding: 0, resource: { buffer: b.positions } },
                { binding: 1, resource: { buffer: b.binOffsets } },
                { binding: 2, resource: { buffer: b.binWriteIndices } },
                { binding: 3, resource: { buffer: b.permutation } },
                { binding: 4, resource: { buffer: b.simParams } },
            ],
        });

        this.bindGroups.applyPerm = d.createBindGroup({
            layout: this.layouts.applyPerm,
            entries: [
                { binding: 0, resource: { buffer: b.positions } },
                { binding: 1, resource: { buffer: b.types } },
                { binding: 2, resource: { buffer: b.permutation } },
                { binding: 3, resource: { buffer: b.positionsBinned } },
                { binding: 4, resource: { buffer: b.typesBinned } },
                { binding: 5, resource: { buffer: b.simParams } },
            ],
        });

        this.bindGroups.forward = d.createBindGroup({
            layout: this.layouts.forward,
            entries: [
                { binding: 0, resource: { buffer: b.positionsBinned } },
                { binding: 1, resource: { buffer: b.typesBinned } },
                { binding: 2, resource: { buffer: b.binOffsets } },
                { binding: 3, resource: { buffer: b.speciesParams } },
                { binding: 4, resource: { buffer: b.simParams } },
                { binding: 5, resource: { buffer: b.scalarFields } },
                { binding: 6, resource: { buffer: b.gradE } },
            ],
        });

        this.bindGroups.updatePos = d.createBindGroup({
            layout: this.layouts.updatePos,
            entries: [
                { binding: 0, resource: { buffer: b.positionsBinned } },
                { binding: 1, resource: { buffer: b.gradE } },
                { binding: 2, resource: { buffer: b.simParams } },
            ],
        });
    }

    /**
     * Upload initial particle positions and types.
     */
    uploadParticles(positions, types) {
        this.device.queue.writeBuffer(this.buffers.positions, 0, positions);
        this.device.queue.writeBuffer(this.buffers.types, 0, types);
    }

    /**
     * Upload species parameters.
     */
    uploadSpeciesParams(packedParams) {
        this.device.queue.writeBuffer(this.buffers.speciesParams, 0, packedParams);
    }

    /**
     * Upload simulation parameters uniform.
     */
    uploadSimParams(config) {
        const data = packSimParams(config);
        this.device.queue.writeBuffer(this.buffers.simParams, 0, data);
    }

    /**
     * Update the dt value in sim params.
     */
    updateDt(dt) {
        this.config.dt = dt;
        this.uploadSimParams(this.config);
    }

    /**
     * Run one simulation step (binning + forward + update).
     */
    encodeStep(encoder) {
        const N = this.config.N;
        const workgroups = Math.ceil(N / 256);

        // Clear cell_counts and bin_write_indices
        encoder.clearBuffer(this.buffers.cellCounts);
        encoder.clearBuffer(this.buffers.binWriteIndices);

        // Pass 1: Count cells
        const p1 = encoder.beginComputePass();
        p1.setPipeline(this.pipelines.countCells);
        p1.setBindGroup(0, this.bindGroups.countCells);
        p1.dispatchWorkgroups(workgroups);
        p1.end();

        // Pass 2: Prefix sum
        const p2 = encoder.beginComputePass();
        p2.setPipeline(this.pipelines.prefixSum);
        p2.setBindGroup(0, this.bindGroups.prefixSum);
        p2.dispatchWorkgroups(1);
        p2.end();

        // Pass 3: Compute permutation
        const p3 = encoder.beginComputePass();
        p3.setPipeline(this.pipelines.computePerm);
        p3.setBindGroup(0, this.bindGroups.computePerm);
        p3.dispatchWorkgroups(workgroups);
        p3.end();

        // Pass 4: Scatter
        const p4 = encoder.beginComputePass();
        p4.setPipeline(this.pipelines.applyPerm);
        p4.setBindGroup(0, this.bindGroups.applyPerm);
        p4.dispatchWorkgroups(workgroups);
        p4.end();

        // Pass 5: Forward kernel
        const p5 = encoder.beginComputePass();
        p5.setPipeline(this.pipelines.forward);
        p5.setBindGroup(0, this.bindGroups.forward);
        p5.dispatchWorkgroups(workgroups);
        p5.end();

        // Pass 6: Update positions (modifies positionsBinned in-place)
        const p6 = encoder.beginComputePass();
        p6.setPipeline(this.pipelines.updatePos);
        p6.setBindGroup(0, this.bindGroups.updatePos);
        p6.dispatchWorkgroups(workgroups);
        p6.end();

        // Copy positionsBinned -> positions for next step's binning
        encoder.copyBufferToBuffer(
            this.buffers.positionsBinned, 0,
            this.buffers.positions, 0,
            N * 8
        );
        // Copy typesBinned -> types
        encoder.copyBufferToBuffer(
            this.buffers.typesBinned, 0,
            this.buffers.types, 0,
            N * 4
        );

        this.stepCount++;
    }

    /**
     * Run multiple steps and submit.
     */
    step(stepsPerFrame = 1) {
        const encoder = this.device.createCommandEncoder();
        for (let s = 0; s < stepsPerFrame; s++) {
            this.encodeStep(encoder);
        }
        this.device.queue.submit([encoder.finish()]);
    }

    /**
     * Read back scalar fields and grad_E for testing.
     * Returns: { scalarFields: Float32Array(N*4), gradE: Float32Array(N*2) }
     */
    async readBack() {
        const N = this.config.N;
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.scalarFields, 0,
                                   this.buffers.scalarFieldsReadback, 0, N * 16);
        encoder.copyBufferToBuffer(this.buffers.gradE, 0,
                                   this.buffers.gradEReadback, 0, N * 8);
        encoder.copyBufferToBuffer(this.buffers.positionsBinned, 0,
                                   this.buffers.positionsReadback, 0, N * 8);
        this.device.queue.submit([encoder.finish()]);

        await this.buffers.scalarFieldsReadback.mapAsync(GPUMapMode.READ);
        await this.buffers.gradEReadback.mapAsync(GPUMapMode.READ);
        await this.buffers.positionsReadback.mapAsync(GPUMapMode.READ);

        const scalarFields = new Float32Array(
            this.buffers.scalarFieldsReadback.getMappedRange().slice(0)
        );
        const gradE = new Float32Array(
            this.buffers.gradEReadback.getMappedRange().slice(0)
        );
        const positions = new Float32Array(
            this.buffers.positionsReadback.getMappedRange().slice(0)
        );

        this.buffers.scalarFieldsReadback.unmap();
        this.buffers.gradEReadback.unmap();
        this.buffers.positionsReadback.unmap();

        return { scalarFields, gradE, positions };
    }

    /**
     * Read back current particle positions/types in unbinned order.
     * Returns: { positions: Float32Array(N*2), types: Uint32Array(N) }
     */
    async readParticleState() {
        const N = this.config.N;

        const positionsReadback = this.device.createBuffer({
            size: N * 8,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const typesReadback = this.device.createBuffer({
            size: N * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.buffers.positions, 0, positionsReadback, 0, N * 8);
        encoder.copyBufferToBuffer(this.buffers.types, 0, typesReadback, 0, N * 4);
        this.device.queue.submit([encoder.finish()]);

        await positionsReadback.mapAsync(GPUMapMode.READ);
        await typesReadback.mapAsync(GPUMapMode.READ);

        const positions = new Float32Array(positionsReadback.getMappedRange().slice(0));
        const types = new Uint32Array(typesReadback.getMappedRange().slice(0));

        positionsReadback.unmap();
        typesReadback.unmap();
        positionsReadback.destroy();
        typesReadback.destroy();

        return { positions, types };
    }

    /**
     * Run just the binning + forward pass (no position update) for testing.
     */
    async runForwardOnly() {
        const N = this.config.N;
        const workgroups = Math.ceil(N / 256);
        const encoder = this.device.createCommandEncoder();

        encoder.clearBuffer(this.buffers.cellCounts);
        encoder.clearBuffer(this.buffers.binWriteIndices);

        // Count cells
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.countCells);
        pass.setBindGroup(0, this.bindGroups.countCells);
        pass.dispatchWorkgroups(workgroups);
        pass.end();

        // Prefix sum
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.prefixSum);
        pass.setBindGroup(0, this.bindGroups.prefixSum);
        pass.dispatchWorkgroups(1);
        pass.end();

        // Compute permutation
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.computePerm);
        pass.setBindGroup(0, this.bindGroups.computePerm);
        pass.dispatchWorkgroups(workgroups);
        pass.end();

        // Apply permutation
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.applyPerm);
        pass.setBindGroup(0, this.bindGroups.applyPerm);
        pass.dispatchWorkgroups(workgroups);
        pass.end();

        // Forward kernel
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.forward);
        pass.setBindGroup(0, this.bindGroups.forward);
        pass.dispatchWorkgroups(workgroups);
        pass.end();

        this.device.queue.submit([encoder.finish()]);

        return this.readBack();
    }

    destroy() {
        for (const buf of Object.values(this.buffers)) {
            buf.destroy();
        }
    }
}
