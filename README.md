# Viterbi's Algorithm

This project provides a command line interface for solving hidden Markov models (HMMs) using Viterbi's algorithm.

Hidden Markov models are a two-staged stochastic process involving a Markov chain: we first allow our Markov chain to converge to its stationary distribution and then, for each transition, randomly draw a sample from a probability distribution that depends on the current state of the Markov chain. This process can be framed as randomly transitioning between different bags behind a curtain and, each time a new bag is selected, revealing a randomly selected marble from the current bag. 

Naively, determining the most-likely sequence of bags visited takes an exponential number of computations; however, it can be solved in a polynimial number of computations using Viterbi's algorithm.

The source code for the HMM data structure is located in `src/viterbi/hidden_markov_model.py` and the source code for Viterbi's algorithm is located in `src/viterbi/viterbi_algorithm.py`.

## Installation

### Downloading the Repository

The code for this project is stored on GitHub. To download this repository, you may either [download it as a ZIP file](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives#downloading-source-code-archives) or run the following command with [Git](https://git-scm.com):
```bash
git clone https://github.com/jackczen/viterbi.git
```

### Building the Docker Image
This project relies on [Docker](https://www.docker.com/) for managing application dependencies. 
If you do not have Docker installed, you can get it [here](https://docs.docker.com/engine/install/).

One Docker is installed, you can navigate to this project's main directory and then build the `viterbi` Docker image using:
```bash
docker build -t viterbi .
```

## Usage

### Command Line Interface (CLI)

To solve a hidden Markov model, run the `viterbi` Docker image using:
```bash
docker run --rm viterbi solve MAT SAMPLING SEQUENCE
```
where

| Argument | Meaning | Syntax                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------- | ------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MAT`      | The HMM's transition matrix. | Write `MAT` as a quoted two-dimensional list in Python-like syntax, i.e., `"[[0.4, 0.6], [0.2, 0.8]]"`. This matrix must be square and stochastic (all rows run to $1$). `MAT[i][j]` will be interpreted as the probability of transitioning from bag $i$ to bag $j$.                                                                                                                                                        |
| `SAMPLING` | Distributions for sampling marbles, given each bag. | Write `SAMPLING` as a quoted two-dimensional list in Python-like syntax, i.e., `"[[0.1, 0.9], [0.35, 0.65]]"`. This array must have shape $n \times m$, where $n$ is the number of states ("bags") and $m$ is the number of "marbles." `SAMPLING[i][j]` will be interpreted as the probability of drawing marble $j$, given we are in bag $i$, i.e., $\Pr(\textrm{marble } j \mid \textrm{bag }i)$. The entries of `SAMPLING[i]` must sum to $1$ for every bag $i$. |
| `SEQUENCE` | Observed sequence of marbles. | Write `SEQUENCE` as a quoted one-dimensional list in Python-like syntax, i.e., `"[0, 1, 0, 0, 1, 1]"`. This array can have any length; however, each list element $e$ must satisfy $0 \leq e < m$ where $m$ is the number of "marbles" used in constructing `SAMPLING`. `SEQUENCE[i]` is interpreted as observing marble `SEQUENCE[i]` as the $i$-th marble (where the $0$-th marble is the first marble observed).                                 |

## Large Example

### Problem Statement

Suppose we have the transition matrix:
$$T=\begin{pmatrix}
    0.1 & 0.2 & 0.3 & 0.2 & 0.2 \\
    0.6 & 0.1 & 0.1 & 0.1 & 0.1 \\
    0.2 & 0.1 & 0.1 & 0.5 & 0.1 \\
    0.2 & 0.2 & 0.2 & 0.2 & 0.2 \\
    0.4 & 0.1 & 0.1 & 0.1 & 0.3
\end{pmatrix}$$
with state space $\Omega = \{A, B, C, D, E\}$ for bags $A,B,C,D,E$ all containing red, orange, yellow, green, blue, and purple marbles. We know:

* bag $A$ contains 1 red marbles, 2 orange marbles, 1 yellow marbles, 2 green marbles, 3 blue marbles, and 1 purple marbles.
* bag $B$ contains 1 red marbles, 1 orange marbles, 1 yellow marbles, 2 green marbles, 1 blue marbles, and 4 purple marbles.
* bag $C$ contains 1 red marbles, 3 orange marbles, 1 yellow marbles, 1 green marbles, 3 blue marbles, and 1 purple marbles.
* bag $D$ contains 3 red marbles, 1 orange marbles, 1 yellow marbles, 1 green marbles, 2 blue marbles, and 2 purple marbles.
* bag $E$ contains 4 red marbles, 1 orange marbles, 2 yellow marbles, 1 green marbles, 1 blue marbles, and 1 purple marbles.

Suppose we observe the following sequence of 125 marbles: 

> orange, purple, red, green, purple, yellow, red, purple, green, red, yellow, blue, green, red, green, orange, purple, red, red, purple, yellow, blue, green, yellow, purple, red, purple, red, orange, yellow, yellow, blue, orange, purple, orange, purple, red, red, orange, yellow, yellow, green, orange, purple, yellow, yellow, red, purple, orange, purple, yellow, orange, orange, orange, red, red, purple, green, purple, blue, purple, orange, yellow, purple, orange, orange, yellow, orange, orange, red, green, yellow, purple, purple, blue, yellow, orange, orange, blue, purple, yellow, red, yellow, red, yellow, green, red, blue, green, purple, green, purple, blue, green, orange, orange, red, yellow, orange, red, blue, blue, blue, blue, blue, red, green, orange, orange, blue, blue, orange, red, yellow, yellow, yellow, yellow, blue, purple, orange, red, red, orange, green, yellow

What is the most likely sequence of bags visited?

### Remarks on Algorithmic Runtime

Brute-forcing the solution to this problem takes, in general, $O(n^m)$ steps, where $n$ is the number of bags and $m$ is the length of the sequence observed. This is due to the fact that, for each new marble observed, we must consider $n$ new paths for every path that we've already considered up until this point, meaning the total number of paths searched grows by a factor of $n$ each iteration. *This* instance of the problem would then take approximately $5^{125} \approx 2.35 \times 10^{87}$ computations, which far exceeds the number of atoms in the known universe. 

Conversely, Viterbi's algorithm takes, in general, $O(m \cdot n^2)$ steps, as, for each new marble observed, we only need to consider $n^2$ possible transitions. This follows from the fact that, each time we observe a new marble, we must consider $n$ possible ways (one for each of the most-likely paths from the previous iteration) to add each of the $n$ bags. This means we can solve this instance of the problem in approximately $125 \cdot 5^2 = 3125$ computations using Viterbi's algorithm.

### Solution

To solve this problem with the CLI, we must first encode both the bags and the marbles in our problem as integers:
* encode bags as $A \mapsto 0$, $B \mapsto 1$, $C \mapsto 2$, $D \mapsto 3$, and $E \mapsto 4$.
* encode marbles as $\mathrm{red} \mapsto 0$, $\mathrm{orange} \mapsto 1$, $\mathrm{yellow} \mapsto 2$, $\mathrm{green} \mapsto 3$, $\mathrm{blue} \mapsto 4$, and $\mathrm{purple} \mapsto 5$.

Using the syntax for `MAT`, `SAMPLING`, and `SEQUENCE` from the CLI instructions, we then get that:
```text
MAT = "[[0.1, 0.2, 0.3, 0.2, 0.2], [0.6, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2], [0.4, 0.1, 0.1, 0.1, 0.3]]"
SAMPLING = "[[0.1, 0.2, 0.1, 0.2, 0.3, 0.1], [0.1, 0.1, 0.1, 0.2, 0.1, 0.4], [0.1, 0.3, 0.1, 0.1, 0.3, 0.1], [0.3, 0.1, 0.1, 0.1, 0.2, 0.2], [0.4, 0.1, 0.2, 0.1, 0.1, 0.1]]"
SEQUENCE = "[1, 5, 0, 3, 5, 2, 0, 5, 3, 0, 2, 4, 3, 0, 3, 1, 5, 0, 0, 5, 2, 4, 3, 2, 5, 0, 5, 0, 1, 2, 2, 4, 1, 5, 1, 5, 0, 0, 1, 2, 2, 3, 1, 5, 2, 2, 0, 5, 1, 5, 2, 1, 1, 1, 0, 0, 5, 3, 5, 4, 5, 1, 2, 5, 1, 1, 2, 1, 1, 0, 3, 2, 5, 5, 4, 2, 1, 1, 4, 5, 2, 0, 2, 0, 2, 3, 0, 4, 3, 5, 3, 5, 4, 3, 1, 1, 0, 2, 1, 0, 4, 4, 4, 4, 4, 0, 3, 1, 1, 4, 4, 1, 0, 2, 2, 2, 2, 4, 5, 1, 0, 0, 1, 3, 2]"
```
*Note*, these matrices must be surrounded by quoations as a requirement for bash command line arguments that contain brackets `[` or `]`. To run the solve, we then run:
```bash
docker run --rm viterbi solve "[[0.1, 0.2, 0.3, 0.2, 0.2], [0.6, 0.1, 0.1, 0.1, 0.1], [0.2, 0.1, 0.1, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2], [0.4, 0.1, 0.1, 0.1, 0.3]]" "[[0.1, 0.2, 0.1, 0.2, 0.3, 0.1], [0.1, 0.1, 0.1, 0.2, 0.1, 0.4], [0.1, 0.3, 0.1, 0.1, 0.3, 0.1], [0.3, 0.1, 0.1, 0.1, 0.2, 0.2], [0.4, 0.1, 0.2, 0.1, 0.1, 0.1]]" "[1, 5, 0, 3, 5, 2, 0, 5, 3, 0, 2, 4, 3, 0, 3, 1, 5, 0, 0, 5, 2, 4, 3, 2, 5, 0, 5, 0, 1, 2, 2, 4, 1, 5, 1, 5, 0, 0, 1, 2, 2, 3, 1, 5, 2, 2, 0, 5, 1, 5, 2, 1, 1, 1, 0, 0, 5, 3, 5, 4, 5, 1, 2, 5, 1, 1, 2, 1, 1, 0, 3, 2, 5, 5, 4, 2, 1, 1, 4, 5, 2, 0, 2, 0, 2, 3, 0, 4, 3, 5, 3, 5, 4, 3, 1, 1, 0, 2, 1, 0, 4, 4, 4, 4, 4, 0, 3, 1, 1, 4, 4, 1, 0, 2, 2, 2, 2, 4, 5, 1, 0, 0, 1, 3, 2]"
```
This gives the console output:
```text
...

INFO:src.viterbi.viterbi_algorithm:Selected 1 -> 0 with p=3.2271620413477303e-140
INFO:src.viterbi.viterbi_algorithm:Selected 0 -> 1 with p=8.067905103369326e-141
INFO:src.viterbi.viterbi_algorithm:Selected 0 -> 2 with p=1.2101857655053987e-140
INFO:src.viterbi.viterbi_algorithm:Selected 2 -> 3 with p=2.0169762758423313e-140
INFO:src.viterbi.viterbi_algorithm:Selected 0 -> 4 with p=1.6135810206738651e-140
INFO:src.viterbi.viterbi_algorithm:Vector after marble 125: [3.22716204e-140 8.06790510e-141 1.21018577e-140 2.01697628e-140
 1.61358102e-140]
INFO:__main__:Most likely path: [2, 3, 4, 0, 1, 0, 3, 1, 0, 4, 4, 0, 4, 4, 0, 2, 3, 4, 4, 4, 4, 0, 1, 0, 1, 0, 1, 0, 2, 3, 4, 0, 2, 3, 2, 3, 4, 4, 0, 4, 4, 0, 2, 3, 4, 4, 4, 1, 0, 1, 0, 2, 0, 2, 3, 3, 1, 0, 1, 0, 1, 0, 3, 1, 0, 2, 3, 0, 2, 3, 0, 2, 3, 1, 0, 4, 0, 2, 3, 1, 0, 4, 4, 4, 4, 0, 4, 4, 0, 1, 0, 1, 0, 1, 0, 2, 3, 4, 0, 4, 0, 2, 3, 0, 2, 3, 1, 0, 2, 3, 0, 2, 3, 4, 4, 4, 4, 0, 1, 0, 4, 4, 0, 1, 0]
```
This means that the most-likely sequence of bags visited is $C$, $D$, $E$, $A$, $B$, $A$, $D$, $B$, $A$, $E$, $E$, $A$, $E$, $E$, $A$, $C$, $D$, $E$, $E$, $E$, $E$, $A$, $B$, $A$, $B$, $A$, $B$, $A$, $C$, $D$, $E$, $A$, $C$, $D$, $C$, $D$, $E$, $E$, $A$, $E$, $E$, $A$, $C$, $D$, $E$, $E$, $E$, $B$, $A$, $B$, $A$, $C$, $A$, $C$, $D$, $D$, $B$, $A$, $B$, $A$, $B$, $A$, $D$, $B$, $A$, $C$, $D$, $A$, $C$, $D$, $A$, $C$, $D$, $B$, $A$, $E$, $A$, $C$, $D$, $B$, $A$, $E$, $E$, $E$, $E$, $A$, $E$, $E$, $A$, $B$, $A$, $B$, $A$, $B$, $A$, $C$, $D$, $E$, $A$, $E$, $A$, $C$, $D$, $A$, $C$, $D$, $B$, $A$, $C$, $D$, $A$, $C$, $D$, $E$, $E$, $E$, $E$, $A$, $B$, $A$, $E$, $E$, $A$, $B$, $A$.
