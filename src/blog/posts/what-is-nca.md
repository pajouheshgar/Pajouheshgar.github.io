---
layout: layouts/post.njk
title: "What Is a Neural Cellular Automaton?"
date: 2024-08-01
tags: ["nca", "artificial-life", "intro"]
description: "A gentle introduction to Neural Cellular Automata — what they are, why they're interesting, and what questions they open up."
---

Neural Cellular Automata (NCA) are a class of models that sit at a strange and beautiful intersection: they are *cellular automata* in spirit — local, parallel, decentralized — but their update rules are *learned*, not hand-crafted.

## The Classic Idea

A cellular automaton is simple to define. You have a grid of cells, each carrying some state. At each timestep, every cell looks at its neighbors and updates its state according to a fixed rule. Conway's Game of Life is the canonical example: cells are either alive or dead, and three simple rules determine the next state.

What makes CA fascinating is that global structure emerges from purely local interactions. No cell knows about the whole pattern. Yet patterns grow, replicate, and die.

## Adding Neural Networks

The NCA twist: instead of a hand-designed rule, we *learn* the update function. Each cell runs a small neural network that takes in the neighborhood state and outputs the next state. The network is the same for every cell — there is no "global controller."

This is powerful for two reasons:

1. **Expressiveness.** Neural networks can approximate complex update rules that would be impossible to design by hand.
2. **Differentiability.** Because the update rule is a neural network, we can train it end-to-end with gradient descent. We can specify a *target behavior* and let the system figure out what local rules produce it.

## What Can You Train NCA to Do?

- **Texture synthesis** — train a NCA to regenerate a target texture from any starting state [^1]
- **Morphogenesis** — train cells to grow into a target shape
- **Dynamic textures** — train to reproduce the statistics of a moving texture (fire, water, clouds)
- **3D surfaces** — extend to mesh graphs and synthesize textures on arbitrary geometry

The key insight in all of these: the "program" is encoded in the weights of the shared update network, not in any global memory or coordinator.

## Open Questions

This is where it gets interesting. Some things I think about:

- **What is the computational complexity class of NCA?** They are Turing-complete in the classical sense, but with a *learned* rule, what can you express within a fixed network size?
- **Can NCA exhibit self-organized criticality?** Classic SOC systems (sandpiles, forest fires) are at the edge of order and chaos. Can a NCA be trained to live there?
- **What happens on curved spaces?** MeshNCA works on flat-ish meshes. On a hyperbolic surface, neighborhoods are much larger at every scale — does the same architecture still self-organize?

These aren't rhetorical questions — they're things I'm actively thinking about. If any of them resonate with you, [reach out](mailto:e.pajouheshgar@gmail.com).

[^1]: The original Growing Neural Cellular Automata paper (Mordvintsev et al., 2020) demonstrated this beautifully for image textures.
