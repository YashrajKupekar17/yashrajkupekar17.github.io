---
layout: post
title: The Intuition Behind Gradient Descent
date: 2025-12-26 13:50 +0530
math: true
description : Understand GD and its different varients 

---



---

## Introduction

Imagine you have a simple neural network — just a single sigmoid neuron with one input:

<!-- Block math -->

$$
f(x) = \frac{1}{1 + e^{-(w x + b)}}
$$

It has only **two parameters**, $$ w $$ and $$ b $$.  
Now suppose we have a small dataset, and our goal is to make this neuron fit the data well.

So the big question becomes:

> **How do we find the right values of $$ w $$ and $$ b $$?**

The most naïve idea would be:  
“Try some values randomly, calculate the loss, and keep adjusting.”

But this is basically **brute force** — slow, inefficient, and impossible when parameters become large.

A slightly smarter idea might be:  
“Change $$ w $$ a little bit. If loss decreases, keep going in that direction.  
If loss increases, reverse the direction.”

But this “smart guesswork” still fails because:

- the loss surface could be curved,
- parameters interact,
- infinite possible values exist,
- trying every direction is impossible.

So we need something more systematic.

---

## Thinking in Terms of an Error Surface

Instead of randomly trying values, imagine the loss for every possible $$ (w, b) $$ combination forms a 3D surface — a landscape of hills and valleys.  
Our goal is simply to reach the **lowest valley** — the point of minimum loss.

But here’s the challenge:

We cannot compute the entire surface — it's infinite.

Yet, if we assume this surface exists, we can ask a more meaningful question:

> **If I'm standing at some point on this surface (some values of $$ w $$ and $$ b $$), how do I know which direction moves me downhill?**

This is where **gradients** come in.

---

## The Key Insight: Gradients

The gradient of the loss with respect to the parameters tells us:

- **which direction makes the loss increase the fastest**
- therefore, going in the **opposite direction** decreases the loss fastest

This gives us a principled update rule:

<!-- Equation numbering -->

$$
\begin{equation}
w_{t+1} = w_t - \eta \frac{\partial L}{\partial w}
\label{eq:weight_update}
\end{equation}
$$

$$
\begin{equation}
b_{t+1} = b_t - \eta \frac{\partial L}{\partial b}
\label{eq:bias_update}
\end{equation}
$$

This simple rule is known as **Gradient Descent** — the fundamental optimization algorithm behind neural networks.

With it, we finally have a reliable way to move through the error surface, step by step, towards a minimum.

---

## Why Gradient Descent Sometimes Fails

In the previous section, we saw how Gradient Descent (GD) moves along the negative gradient to decrease the loss.  
But GD has certain weaknesses.

### Observation #1 — Slow on Gentle Slopes

When the loss surface becomes flat or has gentle curves:

- gradients are very small
- the parameter updates become tiny
- GD moves painfully slowly

This is why training sometimes appears to “plateau”.

### Observation #2 — Zig-Zag in Narrow Valleys

Sometimes the loss surface looks like a long, curved valley:

Here:

- one direction is steep → gradients are large
- the other direction is flat → gradients are small

GD tends to oscillate left–right along the steep walls instead of smoothly sliding forward.

### Can We Do Better Than Vanilla GD?

Yes — by giving GD **momentum**.

---

## Momentum-Based Gradient Descent

### Intuition

If you are repeatedly pushed in the same direction, you naturally start gaining speed.  
Similarly, if the gradient continues pointing in a consistent direction, we want to take **bigger steps**.

This is like a ball rolling downhill.  
Even if the slope becomes flat, the ball continues moving because it carries **momentum**.

---

### Momentum Update Rule

We introduce a new variable $$ u_t $$ that stores “velocity”:

$$
u_t = \beta u_{t-1} + \nabla w_t
$$

$$
w_{t+1} = w_t - \eta u_t
$$

Where:

- $$ u_t $$ = velocity (exponentially weighted average of past gradients)
- $$ 0 \le \beta < 1 $$ = momentum factor
- typical value: $$ \beta = 0.9 $$

Initial values:

$$
u_{-1} = 0, \quad w_0 = \text{random}
$$

#### Interpretation

This formula means:

- part of your new direction comes from the current gradient
- part of it comes from all past gradients
- weighted exponentially

So momentum is:

$$
u_t = \sum_{\tau=0}^{t} \beta^{t-\tau} \nabla w_\tau
$$

i.e., an **exponentially decaying history of gradients**.

---

### Why Momentum Helps

- ✔ Fast on gentle slopes  
- ✔ Less zig-zag in narrow valleys  
- ✔ Faster convergence than vanilla GD  

Often 3× to 10× improvement in practice.

---

## But Momentum Has a Problem: Overshooting

Momentum can become too aggressive.

Imagine a narrow valley shaped like a deep “U”.  
Momentum makes you roll into the valley fast — but can also overshoot and climb up the other side:

- oscillating in and out
- taking several U-turns
- wasting time correcting the path

Even though momentum eventually converges, the oscillations can be large.

---

## Nesterov Accelerated Gradient (NAG)  
*“Look Before You Leap”*

### Key Idea

Momentum blindly follows the velocity.  
NAG tries to be smarter.

Before computing the gradient, NAG **looks ahead** to where the momentum is already taking it:

$$
u_t = \beta u_{t-1} + \nabla \left( w_t - \beta u_{t-1} \right)
$$

$$
w_{t+1} = w_t - \eta u_t
$$

### Why This Helps

- NAG computes gradient at the **future position**, not the current one
- This gives a better estimate of the direction
- NAG corrects itself early and avoids dramatic overshooting

### Benefits

- ✔ Fewer oscillations  
- ✔ Faster convergence  
- ✔ More stable than momentum  

NAG is widely used when training deep networks from scratch.

---

## Batch, Stochastic, and Mini-Batch Gradient Descent

Before going further, let’s revisit the three flavors of gradient descent.

### 1. Batch Gradient Descent

- Computes gradient over the entire dataset  
- Stable & accurate  
- Very slow for large datasets  
- Used rarely today due to inefficiency  

### 2. Stochastic Gradient Descent (SGD)

- Updates parameters for **each data point**  
- Very fast updates  
- Very noisy path  
- Loss does not always decrease each step  

SGD helps escape local minima due to noise.

### 3. Mini-Batch Gradient Descent

This is the **standard** method used in deep learning.

- updates using small batches (e.g., 32, 64)  
- better gradient estimate than SGD  
- much faster than batch GD  
- smoother than pure SGD  

---

## Adding Momentum to These Variants

### Mini-Batch Momentum

- combines stability of mini-batches  
- plus speed of momentum  
- extremely effective in practice  

### Mini-Batch NAG

- fastest and most stable in this family  
- used in many classical deep learning setups (CNNs, RNNs)  

---

## Revisiting the Learning Rate Problem

One might think:  
“Why not simply set a higher learning rate so gentle slopes move faster?”

Let’s try:

$$
\eta = 10
$$

### What Happens?

- In gentle slope regions → faster updates  
- In steep slope regions → gradients explode → we overshoot violently  
- The loss may diverge completely  

So:

> **We need a learning rate that adjusts according to the geometry of the surface.**

This brings us to **Learning Rate Schedules** and **Adaptive Learning Rate Methods**.
