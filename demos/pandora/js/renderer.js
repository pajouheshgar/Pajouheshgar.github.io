/**
 * WebGPU renderer for particle visualization.
 */

import { PALETTE } from './species.js';

export class LeniaRenderer {
    constructor() {
        this.pipeline = null;
        this.boundsPipeline = null;
        this.bindGroup = null;
        this.device = null;
        this.context = null;
        this.format = null;
        this.pan = [0, 0];
        this.zoom = 1.0;
        this.pointSize = 3.0;
    }

    async init(device, canvas, simBuffers, config) {
        this.device = device;
        this.N = config.N;
        this.boxSize = config.boxSize;

        // Configure canvas context
        this.context = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device,
            format: this.format,
            alphaMode: 'premultiplied',
        });

        // Load render shader
        const shaderSrc = await fetch('shaders/render.wgsl').then(r => r.text());
        const module = device.createShaderModule({ code: shaderSrc });

        // Render params uniform buffer
        this.renderParamsBuffer = device.createBuffer({
            size: 32, // 8 floats
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Palette buffer
        const paletteData = new Float32Array(10 * 4);
        for (let i = 0; i < 10; i++) {
            paletteData[i * 4 + 0] = PALETTE[i][0];
            paletteData[i * 4 + 1] = PALETTE[i][1];
            paletteData[i * 4 + 2] = PALETTE[i][2];
            paletteData[i * 4 + 3] = 1.0;
        }
        this.paletteBuffer = device.createBuffer({
            size: paletteData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.paletteBuffer, 0, paletteData);

        // Bind group layout
        const bgl = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            ],
        });

        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            vertex: {
                module,
                entryPoint: 'vs_main',
            },
            fragment: {
                module,
                entryPoint: 'fs_main',
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one',
                            operation: 'add',
                        },
                    },
                }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this.boundsPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            vertex: {
                module,
                entryPoint: 'bounds_vs',
            },
            fragment: {
                module,
                entryPoint: 'bounds_fs',
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add',
                        },
                    },
                }],
            },
            primitive: { topology: 'line-strip' },
        });

        // Create bind group using sim buffers
        this.bindGroup = device.createBindGroup({
            layout: bgl,
            entries: [
                { binding: 0, resource: { buffer: simBuffers.positionsBinned } },
                { binding: 1, resource: { buffer: simBuffers.typesBinned } },
                { binding: 2, resource: { buffer: this.renderParamsBuffer } },
                { binding: 3, resource: { buffer: this.paletteBuffer } },
            ],
        });
    }

    updateRenderParams(canvas) {
        const aspect = canvas.width / canvas.height;
        const data = new Float32Array([
            this.pan[0], this.pan[1],
            this.zoom,
            aspect,
            this.pointSize,
            this.boxSize,
            0, 0, // padding
        ]);
        this.device.queue.writeBuffer(this.renderParamsBuffer, 0, data);
    }

    render(encoder, canvas) {
        this.updateRenderParams(canvas);

        const textureView = this.context.getCurrentTexture().createView();
        const renderPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(6, this.N); // 6 vertices per quad, N instances

        // Draw the periodic world boundary square.
        renderPass.setPipeline(this.boundsPipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(5, 1);
        renderPass.end();
    }

    destroy() {
        this.renderParamsBuffer?.destroy();
        this.paletteBuffer?.destroy();
    }
}
