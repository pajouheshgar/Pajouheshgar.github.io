---
layout: layouts/post.njk
title: "Pandora: Scaling Particle Lenia to Millions of Particles"
date: 2026-04-10
tags: ["artificial-life", "particle-lenia", "webgpu", "simulation"]
description: "Pandora is a WebGPU-accelerated Particle Lenia simulator that scales to millions of particles and supports multiple interacting species — and a first attempt to ask whether artificial life can show qualitatively new behavior at larger scales."
references:
  - authors: "Bert Wang-Chak Chan"
    title: "Lenia: Biology of Artificial Life"
    venue: "Complex Systems"
    year: 2019
    url: "https://arxiv.org/abs/1812.05433"
  - authors: "Alexander Mordvintsev, Eyvind Niklasson, Ettore Randazzo"
    title: "Particle Lenia and the Transition to Life"
    venue: "NeurIPS 2022 Workshop on ML for Physical Sciences"
    year: 2022
    url: "https://google-research.github.io/self-organising-systems/particle-lenia/"
  - authors: "Erwan Plantec, Gautier Hamon, Mayalen Etcheverry, Pierre-Yves Oudeyer, Clément Moulin-Frier, Bert Wang-Chak Chan"
    title: "Flow-Lenia: Towards Open-Ended Evolution in Cellular Automata Through Mass Conservation and Parameter Localization"
    venue: "ALIFE 2023"
    year: 2023
    url: "https://arxiv.org/abs/2212.07906"
  - authors: ""
    title: Life in Life
    url: "https://www.youtube.com/watch?v=xP5-iIeKXE8"
---

<div class="demo-embed">
  <iframe src="/demos/pandora/" title="Pandora — interactive Particle Lenia simulation" allowfullscreen></iframe>
  <p class="demo-caption">
    Interactive Pandora simulation — requires a browser with WebGPU support (Chrome 113+, Edge 113+).
    <a href="/demos/pandora/" target="_blank" rel="noopener">Open full screen ↗</a>
  </p>
</div>

## Introduction

Artificial life (ALife) is the study of systems that exhibit life-like behavior — growth, self-organization, reproduction, adaptation — through entirely computational means. The appeal is obvious: if we can distill the *logic* of life into a set of simple rules and watch richness emerge, we learn something deep about what life fundamentally is.

The canonical starting point is **Conway's Game of Life** (1970). A 2D grid of binary cells evolves by three rules — survive with 2–3 live neighbors, die otherwise, birth with exactly 3 — and from these rules alone emerge gliders, oscillators, self-replicators, and eventually structures capable of universal computation. The key insight is that *global complexity can arise from purely local, uniform rules*. No cell knows about the whole pattern.

**Lenia**{% cite 1 %} (Chan, 2019) extends this idea to the continuous domain. Cells hold real values rather than bits, the neighborhood kernel is a smooth ring, and the update rule is a continuous growth function. The result is a rich zoology of smooth, organic-looking creatures — blobs, worms, spirals — that live and move continuously through the grid. Lenia feels closer to actual biology: there is no sharp on/off threshold, and creatures morph gracefully as parameters change.

**Particle Lenia**{% cite 2 %} (Mordvintsev et al., 2022) takes the next step: abandon the fixed grid entirely. Instead of cells at fixed positions sensing their neighbors, free-roaming *particles* each create a field around themselves and respond to the fields created by others. The result is a Lagrangian[^lagr] formulation of Lenia that produces similarly organic patterns — clusters, rings, worms — but now the entities are not anchored to any grid.

[^lagr]: In fluid mechanics and field theories, the *Eulerian* view tracks a fixed point in space and asks what passes through it; the *Lagrangian* view follows individual parcels as they move. Grid-based systems like Lenia are naturally Eulerian — spatial location is the index. Particle systems are Lagrangian — each particle carries a persistent identity, so attaching per-particle properties (species, orientation, memory) is trivial.

Pandora is our implementation of Particle Lenia extended along two axes: **multiple interacting species** and **massive scale** (up to millions of particles). The source code — including a WebGPU implementation (what runs in the demo above) and a CUDA + PyTorch implementation for research use — is available at [github.com/TheDevilWillBeBee/Pandora](https://github.com/TheDevilWillBeBee/Pandora).

## Motivation: Multiple Species

One of the aesthetically unsatisfying aspects of Lenia is that its creatures are *solitary*. Each creature lives in its own parameter space — the specific choice of kernel and growth function that gives rise to it. Two Lenia creatures with different parameters cannot coexist in the same grid; the update rule is uniform across the grid, so there is no notion of "creature A follows rules for species 1 and creature B follows rules for species 2."

There have been attempts to address this. **FlowLenia**{% cite 3 %} allows the Lenia parameters themselves to flow and vary spatially through the grid, enabling something like co-evolution. But this is fundamentally still an Eulerian approach: parameters live at grid positions, not with individual entities.

Particle Lenia offers a more natural solution. Because each particle has a persistent identity — it is *the same particle* from one timestep to the next — we can attach a **type** (species) to it as a permanent attribute. Different species can carry different parameters and behavior. The rules are still entirely local and uniform *within a species*, but inter-species interactions are parameterized independently.

This is much harder to achieve in a grid-based (Eulerian) system, because a grid cell has no persistent identity: what it "is" changes every timestep as new values flow through it.[^euler_lagr]

[^euler_lagr]: This distinction matters for a subtle reason in the math. In single-species Particle Lenia, the U-field is ambiguous — it can be read as either the field that a particle *creates* around itself, or as the particle's *sensitivity* to nearby density. Both interpretations give identical dynamics when there is only one species. In a multi-species system, we must be explicit: **U is the field a particle creates around itself**. A particle of species A creates a field shaped by A's kernel parameters; a particle of species B senses that field with B's own growth function.

## Motivation: The Scaling Question

Almost no artificial life systems have been studied seriously across scale. We run them with hundreds or thousands of particles, observe interesting patterns, and stop. But consider what the physical world looks like:

- Sub-atomic particles → atoms and simple molecules → proteins and DNA → cells → organisms → ecosystems → ....

At each level of organization we find qualitatively *new* emergent phenomena that are not visible at smaller scales. A single water molecule does not have surface tension. A single neuron does not support cognition.

For almost every dynamical system we know of — particle simulations, cellular automata, differential equations — running with more particles or on a larger grid, or simulating for longer timesteps, does not produce qualitatively different behavior. The patterns scale up or repeat proportionally; there is no emergence of new structure at new scales. This motivates us to ask the following question:

**Is there a dynamical system whose microscopic rules are fixed, yet which displays genuinely new qualitative behavior as we increase the simulation scale (temporal and spatial)?**

The only known exceptions are Turing-complete systems like Game of Life, where carefully engineered initial conditions can produce intricate behavior — for example, simulating Game of Life inside Game of Life {% cite 4 %}. However, this is not spontaneous emergence; it requires precisely crafted initial states and relies on the fact that Game of Life is Turing-complete.

More fundamentally, we do not know whether emergent scaling behavior requires a system to be Turing-complete and to rely on elaborately engineered initial conditions, or whether simpler rules can yield qualitatively new phenomena at larger scales simply through the dynamics themselves. With this question in mind, we scale Particle Lenia to millions of particles and observe what, if anything, changes.

## The Math Behind Pandora

Pandora follows the Particle Lenia energy-based formulation. Here is the key structure.

**The U-field.** Each particle $i$ of species $s_i$ creates a ring-shaped scalar field around itself:

$$U_i(\mathbf{x}) = w_k^{(s_i)} \cdot \exp\!\left(-\left(\frac{\|\mathbf{x} - \mathbf{x}_i\| - \mu_k^{(s_i)}}{\sigma_k^{(s_i)}}\right)^{\!2}\right)$$

where $\mu_k$ is the ring radius, $\sigma_k$ is the ring width, and $w_k$ is a normalization weight. The field is a Gaussian bump peaked at distance $\mu_k$ from the particle — a ring, not a disk.

**Superposition.** The total field sensed by particle $j$ is the sum over all other particles:

$$U(\mathbf{x}) = \sum_{i} U_i(\mathbf{x})$$

The U-field is linear and additive — contributions simply stack. This is analogous to electrostatics: each source particle contributes independently to the total field at any point.

**The growth function.** Each particle does not try to maximize or minimize the field it senses. It has a preferred field value $\mu_g$ (with tolerance $\sigma_g$), and is "happy" when $U \approx \mu_g$. The growth function encodes this:

$$G(u) = \exp\!\left(-\left(\frac{u - \mu_g}{\sigma_g}\right)^{\!2}\right)$$

$G$ is close to 1 when $u \approx \mu_g$ and falls off symmetrically — the particle is neutral about field values very far from its optimum in either direction.

**The energy and update rule.** The total energy of particle $i$ combines a repulsion term and a growth term:

$$E_i = R_i - G(U(\mathbf{x}_i))$$

where $R_i$ is the short-range repulsion:

$$R_i = \sum_{j \neq i} \frac{c_\text{rep}^{(s_j)}}{2}(1 - r_{ij})^2 \cdot \mathbf{1}[r_{ij} < 1]$$

Note that $c_\text{rep}^{(s_j)}$ depends on the species of the *source* particle $j$ — different species can be more or less "hard." Particles move by gradient descent on this energy:

$$\dot{\mathbf{x}}_i = -\nabla_{\mathbf{x}_i} E_i$$

which expands into a sum of per-neighbor gradient contributions. The full derivation is given on the [Particle Lenia project page](https://google-research.github.io/self-organising-systems/particle-lenia/).

## Technical Implementation: Scaling to Millions of Particles

The naive pairwise implementation is $O(N^2)$: every particle checks every other particle. At $N = 10^6$, that is $10^{12}$ pair evaluations per timestep — completely intractable, even on a GPU.

Pandora brings this down to roughly $O(N)$ per step using two ideas.

### Spatial Hashing with a Morton (Z-Order) Curve

The interaction kernel is compactly supported — it is zero beyond a cutoff radius $r_\text{max}$ (practically, a few $\sigma_k$ beyond $\mu_k$). This means each particle only needs to look at nearby particles, not all $N$.

Pandora tiles the domain with a uniform grid of cells of size $\varepsilon \approx r_\text{max}$. Each particle belongs to exactly one cell. To find all neighbors of particle $i$, only the $(2r+1)^2$ neighboring cells need to be checked, where $r = \lceil r_\text{max}/\varepsilon \rceil$ — typically 1 or 2.

The cells are ordered in memory using a **Morton (Z-order) hash**: the 2D cell coordinate $(c_x, c_y)$ is mapped to a 1D index by interleaving the bits of $c_x$ and $c_y$. In the WGSL shader this is the `dilate2d` function:

```wgsl
fn dilate2d(x_in: u32) -> u32 {
    var x = x_in & 0x3FFu;
    x = (x | (x << 16u)) & 0x0000FFFFu;
    // ... bit spreading ...
    x = (x | (x << 1u)) & 0x55555555u;
    return x;
}

fn cell2hash(cell: vec2u) -> u32 {
    return dilate2d(cell.x) | (dilate2d(cell.y) << 1u);
}
```

Why Morton and not a simple row-major index? Because the Morton curve has the property that cells that are **close in 2D are also close in 1D** (in the Z-order sense). When a thread iterates over the $(2r+1)^2$ neighboring cells for a given particle, those cells land near each other in the Morton-indexed buffer. This improves GPU cache locality — the memory accesses for a thread's neighbor scan are more contiguous, which reduces cache misses on the GPU.

Particle binning happens in three GPU passes before each forward step:
1. **Count**: each particle atomically increments its cell's counter (`count_cells.wgsl`).
2. **Prefix sum**: turn the per-cell counts into start offsets (`prefix_sum.wgsl`).
3. **Scatter**: reorder the particle buffer so particles in the same cell are contiguous (`compute_permutation.wgsl`, `apply_permutation.wgsl`).

### Thread-Per-Particle Parallelism

The forward kernel (`forward.wgsl`) is fully data-parallel across particles:

```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= params.N) { return; }
    // ... compute grad_E for particle i ...
}
```

Each GPU thread handles exactly one particle. It reads particle $i$'s position and type, iterates over the neighboring cells in the Morton-indexed buffer, accumulates $U$, $G$, $R$, and $\nabla E$, and writes the result. There is no communication between threads during the forward pass — every particle's gradient is independent.

This design scales straightforwardly to millions of particles: more particles means more threads, and modern GPUs have thousands of cores that can absorb them.

### Browser vs. Research Backends

The demo above runs the WebGPU (WGSL) backend entirely in the browser — no server, no installation. The [GitHub repository](https://github.com/TheDevilWillBeBee/Pandora) also contains a **CUDA implementation** with a **PyTorch integration**, which enables differentiable simulation: gradients can flow back through the simulation steps, making it possible to optimize species parameters directly by gradient descent.

## What's Next

Almost all the most interesting artificial life systems — Game of Life, Lenia, Particle Lenia — are **isotropic**: their rules are invariant under rotation. A particle does not know which direction it is "pointing"; it only knows the magnitude of interactions at each distance.

The next planned extension for Pandora is to go beyond isotropy by giving each particle an **orientation** $\theta_i$. This changes the field kernel from a radially symmetric ring to an anisotropic shape, and adds a new differential equation for how $\theta_i$ evolves — coupling the particle's heading to the fields it creates and senses. This is a small structural change but opens a qualitatively different regime of possible behaviors.

---

The full implementation — WebGPU browser demo, CUDA simulation, and PyTorch integration — is at [github.com/TheDevilWillBeBee/Pandora](https://github.com/TheDevilWillBeBee/Pandora).
