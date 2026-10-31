# NEAT from scratch

A from-scratch Python implementation of **NEAT** (NeuroEvolution of Augmenting
Topologies), the algorithm introduced by Kenneth O. Stanley and Risto
Miikkulainen.

## What is NEAT?

NEAT is a genetic algorithm that *evolves* neural networks instead of training
them with gradient descent. It starts from the simplest possible networks and,
generation after generation, grows their structure (new nodes and connections)
while tuning their weights — keeping the networks that perform best. In short:
it searches for both the shape *and* the weights of a neural network at the same
time.

## This implementation

Everything here is written from scratch using only the Python standard library
for the algorithm itself (no ML frameworks). It implements the core ideas from
the original NEAT paper:

- **Minimal starting topology** — networks begin with inputs connected directly
  to outputs and no hidden nodes.
- **Structural mutations** — add-node and add-connection mutations grow the
  network over time.
- **Weight and bias mutations** — perturb or reset existing parameters.
- **Historical markings (innovation numbers)** — so genes can be matched up
  during crossover.
- **Crossover** between two parents, including optional interspecies mating.
- **Speciation** — genomes are grouped into species by a genomic-distance metric
  (excess genes, disjoint genes, and average weight difference) so that new
  structural innovations get time to optimize.
- **Fitness sharing** and **stagnation handling** so that species that stop
  improving are removed.
- **Champion preservation** for sufficiently large species.
- Networks are kept **strictly feedforward** (cycles are detected and repaired).
- **Save / load** of a population via `pickle`.

Fitness is evaluated in parallel across CPU cores using `multiprocessing`.

## Tech stack

- **Python 3** (developed on 3.12)
- **Standard library only** for the NEAT engine (`random`, `math`,
  `multiprocessing`, `pickle`, `copy`, `collections`)
- **pandas** — used only to summarize experiment results in `main.py`

## Project layout

| File | Purpose |
| --- | --- |
| `neat.py` | The NEAT engine: `Genome`, `Node`, `Edge`, `Specie`, `NEAT`, crossover and genomic distance. |
| `activations.py` | Activation functions (sigmoid, tanh, leaky ReLU). |
| `xor.py` | The XOR task: hyperparameters, fitness function, and genome evaluation. |
| `simulation.py` | A small helper that wires a task into the NEAT engine and reports results. |
| `main.py` | Entry point — runs the XOR experiment and prints a summary. |
| `results.csv` | A sample per-generation training trace from one XOR run. |
| `neat.ipynb` | Exploratory notebook used during development. |

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the XOR experiment

```bash
python main.py
```

This evolves networks to solve XOR — the classic non-linearly-separable problem
that cannot be solved without hidden units. By default it performs 100
independent runs (each up to 1000 generations, using all available CPU cores)
and prints a statistical summary at the end.

To get a quick look instead of the full sweep, lower `runs` (and optionally
`max_generations`) in `xor.py`.

## Results

On XOR the algorithm reliably evolves compact networks that solve the problem,
typically finding a solution with only one or two hidden nodes. Fitness here
measures how close the four XOR outputs are to their targets, so a fitness near
100% means the network has effectively learned XOR.

`results.csv` contains a sample per-generation trace from a single run, which
climbs to roughly 0.97 fitness within a few hundred generations.

## Reference

K. O. Stanley and R. Miikkulainen, *Evolving Neural Networks through Augmenting
Topologies*, Evolutionary Computation, 2002.
