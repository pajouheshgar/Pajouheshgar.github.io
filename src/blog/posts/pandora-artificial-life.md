---
layout: layouts/post.njk
title: "Pandora: Stateful Particle Lenia"
date: 2026-04-10
tags: ["artificial-life", "particle-lenia", "webgpu", "simulation"]
description: "Pandora extends Particle Lenia with a per-particle state vector that modulates every interaction — increasing the variety of patterns and behaviors the system can produce — while scaling to millions of particles with multiple interacting species."
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

*Updated July 2026: Pandora's particles now carry a dynamic state vector $\mathbf{s}$ that modulates all of their interactions — the model described below is this stateful version.*

<div class="demo-embed">
  <iframe src="/demos/pandora/" title="Pandora — interactive Particle Lenia simulation" allowfullscreen></iframe>
  <p class="demo-caption">
    Interactive Pandora simulation — requires a browser with WebGPU support (Chrome 113+, Edge 113+).
    <a href="/demos/pandora/" target="_blank" rel="noopener">Open full screen ↗</a>
  </p>
</div>

**Reading the visualization:** each particle's *color* shows the first three channels of its state vector $\mathbf{s}$ (mapped from $[-1,1]$ to RGB), so color changes as a particle's state evolves. Its *shape* shows its species: circles for species 0, then triangles, squares, pentagons, and so on. The **State dim (D)** slider sets the dimensionality of $\mathbf{s}$ (changing it resets the simulation). The camera slowly tours the world on its own; it pauses as soon as you click, drag, scroll, or press a key, and resumes after a moment of idleness — the **Auto Camera** button in the controls turns it off entirely.

## Introduction

Artificial life (ALife) is the study of systems that exhibit life-like behavior — growth, self-organization, reproduction, adaptation — through entirely computational means. The appeal is obvious: if we can distill the *logic* of life into a set of simple rules and watch richness emerge, we learn something deep about what life fundamentally is.

The canonical starting point is **Conway's Game of Life** (1970). A 2D grid of binary cells evolves by three rules — survive with 2–3 live neighbors, die otherwise, birth with exactly 3 — and from these rules alone emerge gliders, oscillators, self-replicators, and eventually structures capable of universal computation. The key insight is that *global complexity can arise from purely local, uniform rules*. No cell knows about the whole pattern.

**Lenia**{% cite 1 %} (Chan, 2019) extends this idea to the continuous domain. Cells hold real values rather than bits, the neighborhood kernel is a smooth ring, and the update rule is a continuous growth function. The result is a rich zoology of smooth, organic-looking creatures — blobs, worms, spirals — that live and move continuously through the grid. Lenia feels closer to actual biology: there is no sharp on/off threshold, and creatures morph gracefully as parameters change.

**Particle Lenia**{% cite 2 %} (Mordvintsev et al., 2022) takes the next step: abandon the fixed grid entirely. Instead of cells at fixed positions sensing their neighbors, free-roaming *particles* each create a field around themselves and respond to the fields created by others. The result is a Lagrangian[^lagr] formulation of Lenia that produces similarly organic patterns — clusters, rings, worms — but now the entities are not anchored to any grid.

[^lagr]: In fluid mechanics and field theories, the *Eulerian* view tracks a fixed point in space and asks what passes through it; the *Lagrangian* view follows individual parcels as they move. Grid-based systems like Lenia are naturally Eulerian — spatial location is the index. Particle systems are Lagrangian — each particle carries a persistent identity, so attaching per-particle properties (species, orientation, memory) is trivial.

Pandora is our implementation of Particle Lenia extended along three axes: a **per-particle state vector** that modulates every interaction, **multiple interacting species**, and **massive scale** (up to millions of particles). The source code — including a WebGPU implementation (what runs in the demo above) and a CUDA + PyTorch implementation for research use — is available at [github.com/TheDevilWillBeBee/Pandora](https://github.com/TheDevilWillBeBee/Pandora).

## Motivation: Multiple Species

One of the aesthetically unsatisfying aspects of Lenia is that its creatures are *solitary*. Each creature lives in its own parameter space — the specific choice of kernel and growth function that gives rise to it. Two Lenia creatures with different parameters cannot coexist in the same grid; the update rule is uniform across the grid, so there is no notion of "creature A follows rules for species 1 and creature B follows rules for species 2."

There have been attempts to address this. **FlowLenia**{% cite 3 %} allows the Lenia parameters themselves to flow and vary spatially through the grid, enabling something like co-evolution. But this is fundamentally still an Eulerian approach: parameters live at grid positions, not with individual entities.

Particle Lenia offers a more natural solution. Because each particle has a persistent identity — it is *the same particle* from one timestep to the next — we can attach a **type** (species) to it as a permanent attribute. Different species can carry different parameters and behavior. The rules are still entirely local and uniform *within a species*, but inter-species interactions are parameterized independently.

This is much harder to achieve in a grid-based (Eulerian) system, because a grid cell has no persistent identity: what it "is" changes every timestep as new values flow through it.[^euler_lagr]

[^euler_lagr]: This distinction matters for a subtle reason in the math. In single-species Particle Lenia, the U-field is ambiguous — it can be read as either the field that a particle *creates* around itself, or as the particle's *sensitivity* to nearby density. Both interpretations give identical dynamics when there is only one species. In a multi-species system, we must be explicit: **U is the field a particle creates around itself**. A particle of species A creates a field shaped by A's kernel parameters; a particle of species B senses that field with B's own growth function.

## Motivation: A State for Every Particle

Species take the Lagrangian idea one step, but they are *static*: a particle is born a member of species 3 and stays that way forever. Everything else about how two particles interact is determined by their distance alone. Real cells are not like this — two cells of the same type can behave very differently depending on their internal chemical state, and that state itself changes in response to the neighborhood.

Pandora's stateful model pushes the same Lagrangian logic further: alongside its position and species, each particle carries a **state vector** $\mathbf{s}_i$ — a $D$-dimensional unit vector that evolves continuously over time. The state acts as a *soft, dynamic identity*: how strongly particle $j$ affects particle $i$ depends on how aligned their states are. Particles with similar states see each other at full strength; particles with orthogonal or opposing states are mutually invisible, even when they sit right next to each other.

The purpose of adding state is to **increase the variety and number of patterns and behaviors that can arise in the system**. In stateless Particle Lenia, two particles at the same distance always interact identically, which bounds how much structure the system can express. With state, the *effective* interaction network is itself a dynamical variable: groups of particles can differentiate, couple, and decouple over time, and the same spatial configuration can support many different behaviors depending on the state configuration living on top of it. You can think of $\mathbf{s}$ as a continuous, dynamic generalization of species — instead of $M$ discrete types fixed at initialization, there is a continuum of possible identities on the unit sphere, and the dynamics decide which identities form and persist.

## Motivation: The Scaling Question

Almost no artificial life systems have been studied seriously across scale. We run them with hundreds or thousands of particles, observe interesting patterns, and stop. But consider what the physical world looks like:

- Sub-atomic particles → atoms and simple molecules → proteins and DNA → cells → organisms → ecosystems → ....

At each level of organization we find qualitatively *new* emergent phenomena that are not visible at smaller scales. A single water molecule does not have surface tension. A single neuron does not support cognition.

For almost every dynamical system we know of — particle simulations, cellular automata, differential equations — running with more particles or on a larger grid, or simulating for longer timesteps, does not produce qualitatively different behavior. The patterns scale up or repeat proportionally; there is no emergence of new structure at new scales. This motivates us to ask the following question:

**Is there a dynamical system whose microscopic rules are fixed, yet which displays genuinely new qualitative behavior as we increase the simulation scale (temporal and spatial)?**

The only known exceptions are Turing-complete systems like Game of Life, where carefully engineered initial conditions can produce intricate behavior — for example, simulating Game of Life inside Game of Life {% cite 4 %}. However, this is not spontaneous emergence; it requires precisely crafted initial states and relies on the fact that Game of Life is Turing-complete.

More fundamentally, we do not know whether emergent scaling behavior requires a system to be Turing-complete and to rely on elaborately engineered initial conditions, or whether simpler rules can yield qualitatively new phenomena at larger scales simply through the dynamics themselves. With this question in mind, we scale Particle Lenia to millions of particles and observe what, if anything, changes.

## The Math Behind Pandora

Pandora follows the energy-based formulation of Particle Lenia. Each particle $i$ carries three quantities: a position $\mathbf{x}_i \in \mathbb{R}^2$, a fixed species $t_i$ that selects its parameters, and a **state vector**

$$\mathbf{s}_i \in \mathbb{S}^{D-1} \subset \mathbb{R}^D, \qquad \|\mathbf{s}_i\| = 1$$

a unit vector on the $D$-dimensional sphere ($D$ is the "State dim" slider in the demo). At initialization each $\mathbf{s}_i$ is drawn from an isotropic Gaussian and normalized to unit length, so the initial states are uniformly distributed over the sphere. Positions and states both evolve; the species does not.

**The kernel.** Each particle creates a ring-shaped scalar field around itself:

$$K_i(r) = w_k^{(t_i)} \cdot \exp\!\left(-\left(\frac{r - \mu_k^{(t_i)}}{\sigma_k^{(t_i)}}\right)^{\!2}\right)$$

where $\mu_k$ is the ring radius, $\sigma_k$ is the ring width, and $w_k$ is a normalization weight — a Gaussian bump peaked at distance $\mu_k$, a ring rather than a disk.

**The U-field.** The field particle $i$ senses is a superposition of these kernels, but each neighbor's contribution is weighted by how *aligned* the two particles' states are, passed through a ReLU:

$$U_i = K_i(0) + \sum_{j \neq i} \max\!\left(0,\; \langle \mathbf{s}_i, \mathbf{s}_j \rangle\right) K_j(r_{ij})$$

Aligned particles see each other at full strength; particles with $\langle \mathbf{s}_i, \mathbf{s}_j \rangle \le 0$ are mutually invisible. The self-term $K_i(0)$ carries weight $\langle \mathbf{s}_i, \mathbf{s}_i \rangle = 1$. Setting all states equal recovers the stateless multi-species Particle Lenia — every weight becomes 1.

**The growth function.** Each particle does not try to maximize or minimize the field it senses. It has a preferred field value $\mu_g$ (with tolerance $\sigma_g$), and is "happy" when $U \approx \mu_g$:

$$G(u) = \exp\!\left(-\left(\frac{u - \mu_g}{\sigma_g}\right)^{\!2}\right)$$

$G$ is close to 1 when $u \approx \mu_g$ and falls off symmetrically — the particle is neutral about field values very far from its optimum in either direction.

**The energy.** The total energy of particle $i$ combines a short-range repulsion term and the growth term:

$$E_i = R_i - G(U_i), \qquad R_i = \sum_{j \neq i} \frac{c_\text{rep}^{(t_j)}}{2}(1 - r_{ij})^2 \cdot \mathbf{1}[r_{ij} < 1]$$

The repulsion is deliberately state-*independent*, so particles never overlap regardless of their states. (Note that $c_\text{rep}^{(t_j)}$ depends on the species of the *source* particle $j$ — different species can be more or less "hard.")

**The update rule.** Everything evolves by gradient descent on this one energy — the position *and* the state:

$$\dot{\mathbf{x}}_i = -\nabla_{\mathbf{x}_i} E_i, \qquad \dot{\mathbf{s}}_i = -\nabla_{\mathbf{s}_i} E_i$$

with the state gradient taken on the sphere: the raw gradient is projected onto the tangent plane at $\mathbf{s}_i$, and $\mathbf{s}_i$ is renormalized after each Euler step. That is the entire model — one energy, two gradient flows. Because $U_i$ contains the state weights, the position gradient automatically only feels neighbors the particle is aligned with; because $R$ has no state dependence, the state gradient reduces to $G'(U_i)\, \partial U_i / \partial \mathbf{s}_i$. The stateless base model and its expanded per-neighbor gradients are derived on the [Particle Lenia project page](https://google-research.github.io/self-organising-systems/particle-lenia/).

Unpacking the state flow gives a nice intuition: $\partial U_i / \partial \mathbf{s}_i$ points toward the kernel-weighted average of the visible neighbors' states, scaled by the growth derivative $G'(U_i)$. When a particle senses less field than it prefers ($U_i < \mu_g$, so $G' > 0$), it rotates its state *toward* its neighbors' states — aligning with them increases the field it sees. When it is oversaturated ($G' < 0$), it rotates *away*, decoupling itself from the crowd. Particles therefore recruit interaction partners when lonely and shed them when overcrowded, purely as a consequence of the energy descent.

This coupling is what generates the extra richness: the states and the positions co-evolve, and structures can now differ not just in their geometry but in the state configuration painted onto them. In the demo the first three channels of $\mathbf{s}_i$ are rendered as the particle's RGB color, so this differentiation is directly visible — a string or ring with a coherent color is a group of particles that have locked their states together.

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

Each GPU thread handles exactly one particle. It reads particle $i$'s position, type, and state vector, iterates over the neighboring cells in the Morton-indexed buffer, accumulates $U$, $G$, $R$, $\nabla E$, and the state force, and writes the result. There is no communication between threads during the forward pass — every particle's gradient is independent.

The state vectors add $D$ floats per particle that ride along through the whole pipeline: they are reordered together with positions and types in the binning scatter, the forward kernel computes the $D$-dimensional dot product $\langle \mathbf{s}_i, \mathbf{s}_j \rangle$ inside the same neighbor loop that evaluates the kernel, and the position-update pass integrates and renormalizes the states in place. The neighbor scan is unchanged — state weighting reads each neighbor's state exactly once — so the stateful model keeps the same $O(N)$ scaling.

This design scales straightforwardly to millions of particles: more particles means more threads, and modern GPUs have thousands of cores that can absorb them.

### Browser vs. Research Backends

The demo above runs the WebGPU (WGSL) backend entirely in the browser — no server, no installation. The [GitHub repository](https://github.com/TheDevilWillBeBee/Pandora) also contains a **CUDA implementation** with a **PyTorch integration**, which enables differentiable simulation: gradients can flow back through the simulation steps, making it possible to optimize species parameters directly by gradient descent.

## What's Next

Almost all the most interesting artificial life systems — Game of Life, Lenia, Particle Lenia — are **isotropic**: their rules are invariant under rotation. The state vector $\mathbf{s}$ breaks the symmetry between *particles*, but the interactions are still isotropic in *space*: a particle does not know which direction it is "pointing"; it only knows the magnitude of interactions at each distance.

The next planned extension for Pandora is to go beyond spatial isotropy by giving each particle an **orientation** $\theta_i$. This changes the field kernel from a radially symmetric ring to an anisotropic shape, and adds a new differential equation for how $\theta_i$ evolves — coupling the particle's heading to the fields it creates and senses. This is a small structural change but opens a qualitatively different regime of possible behaviors.

---

The full implementation — WebGPU browser demo, CUDA simulation, and PyTorch integration — is at [github.com/TheDevilWillBeBee/Pandora](https://github.com/TheDevilWillBeBee/Pandora).
