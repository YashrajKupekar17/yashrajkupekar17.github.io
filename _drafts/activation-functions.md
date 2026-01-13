---
layout: post
title: Activation Functions
---



There’s a famous result in neural networks called the **Universal Approximation Theorem**. It says that a feedforward network with even a single hidden layer can approximate any continuous function — as long as it uses **non-linear activation functions** and has enough neurons.

The reason is simple.

If you only stack linear transformations,

[
y = w_4(w_3(w_2(w_1 x)))
]

you still end up with a **linear function**. Depth doesn’t help — you can only learn linear decision boundaries. Real problems are rarely linear.
Non-linear activations are what give neural networks their expressive power.

---

## Sigmoid — where it all started (and why we moved on)

The sigmoid function

[
\sigma(x) = \frac{1}{1 + e^{-x}}
]

was the default choice in early neural networks. It maps everything neatly into ((0,1)), and its derivative has a clean form:

[
\sigma'(x) = \sigma(x)(1 - \sigma(x)).
]

But sigmoid comes with serious drawbacks.

### Vanishing gradients

When the output approaches 0 or 1, the neuron **saturates** and the derivative goes to zero. During backpropagation, gradients shrink as they move backward through layers. In deep networks this effectively kills learning — weights stop updating and training stalls.

### Not zero-centered

Sigmoid outputs are always positive. That means gradients in the next layer are biased in one direction. Optimization becomes inefficient: instead of moving smoothly in weight space, gradient descent **zig-zags**, slowing convergence.

### Computational cost

Computing exponentials millions of times per training step isn’t cheap — especially compared to simpler activations.

For all these reasons, sigmoid is now mostly avoided in hidden layers.

---

## Tanh — a small improvement

Tanh fixes one issue: it is **zero-centered**, mapping inputs to ((-1, 1)). That helps optimization.

But it still suffers from:

* saturation,
* vanishing gradients,
* expensive exponential computation.

So while better than sigmoid, it doesn’t solve the core training problems in deep networks.

---

## ReLU — the breakthrough

Then came the simplest idea:

[
f(x) = \max(0, x)
]

ReLU changed everything.

* No saturation for positive values → gradients flow cleanly.
* Extremely cheap to compute.
* In practice, networks train **several times faster** than with sigmoid or tanh.

ReLU made deep learning practical at scale.

### The downside: dying neurons

If a neuron’s weights and bias shift so that its input is always negative, it outputs 0 forever. Worse, the gradient is also 0 — so the neuron never recovers. It’s effectively dead.

Careful initialization and learning rates help, but the problem led to several ReLU variants.

---

## ReLU variants that fix dying neurons

**Leaky ReLU**
[
f(x) = \max(\alpha x, x), \quad \alpha \approx 0.01
]
Keeps a small gradient alive for negative inputs.

**Parametric ReLU**
Same idea, but (\alpha) is learned.

**ELU**
[
f(x)=
\begin{cases}
x & x>0 \
\alpha(e^x-1) & x\le 0
\end{cases}
]
Produces negative outputs, closer to zero-centered activations, but costs more to compute.

**Maxout**
[
f(x)=\max(w_1x+b_1,\dots,w_kx+b_k)
]
Generalizes ReLU and Leaky ReLU, avoids saturation and dead neurons — at the cost of **k× more parameters**.

---

## Modern activations: GELU, SELU, Swish

### GELU

Instead of a hard cutoff, GELU weights inputs probabilistically:

[
\text{GELU}(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0,1)
]

In practice, it’s approximated as:

[
x \cdot \sigma(1.702x)
]

GELU is now the **standard activation in transformers** — used in BERT, GPT, and most modern language models.

---

### SELU

SELU is designed to be **self-normalizing**:

[
f(x)=
\begin{cases}
\lambda x & x>0 \
\lambda\alpha(e^x-1) & x\le 0
\end{cases}
]

With carefully chosen constants (\lambda) and (\alpha), activations naturally maintain zero mean and unit variance across layers — often removing the need for batch normalization.

---

### Swish / SiLU

[
f(x) = x \cdot \sigma(\beta x)
]

When (\beta=1), it’s called **SiLU**. When (\beta) is learnable, it’s **Swish**.
Discovered both by theory and automated search, it often outperforms ReLU in deep networks and is widely used in modern CNNs.

---

## What actually works in practice

* **CNNs** → ReLU or Swish/SiLU
* **Transformers / NLP** → GELU
* **RNN gates** → Sigmoid and tanh (still useful here)
* **Deep fully connected nets** → Leaky ReLU, ELU, or GELU
* **Avoid sigmoid in hidden layers** unless you have a very specific reason.

---

## The real lesson

Activation functions aren’t just mathematical details. They shape:

* how gradients flow,
* how fast models converge,
* whether neurons die,
* and whether deep learning is even feasible.

Sigmoid failed not because it was “wrong,” but because it made optimization unnecessarily hard. ReLU and its successors succeeded because they respect how gradient-based learning actually behaves.

New activations will keep appearing — often discovered by automated search — but the important part isn’t memorizing formulas.

It’s understanding **the problems they solve**:
vanishing gradients, biased updates, dead neurons, and slow convergence.

Once you see that, activation functions stop feeling like magic — and start feeling like engineering.
