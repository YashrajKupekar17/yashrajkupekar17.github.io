---
layout: post
title: Weight initialization
date: 2026-01-08 19:42 +0530
math: true
description: We will dive into some weight initialization techiniques and how each one is good or bad 
---
# Weight Initialization in Deep Learning

*Why the way you start your network decides whether it will ever learn*

When we build a neural network, we usually think about architecture, data, optimizers, loss functions. But long before any of that matters, there’s a quiet decision that already shapes everything: **how the weights are initialized**.

Get this wrong and nothing works.  
Not slowly. Not badly.  
Just… not at all.

Loss doesn’t go down. Gradients vanish. Neurons die. And you end up debugging everything except the real problem — bad initialization.

---

When training starts, two things keep happening.

Data flows forward through the layers.

$$
\text{Input} \rightarrow \text{Layer 1} \rightarrow \text{Layer 2} \rightarrow \dots \rightarrow \text{Output}
$$

Errors flow backward to update the weights.

$$
\text{Output error} \rightarrow \nabla W_L \rightarrow \dots \rightarrow \nabla W_1
$$

For learning to work, both flows have to stay healthy. If activations explode or fade away in the forward pass, or if gradients disappear in the backward pass, training stops.

And the very first thing that decides whether these flows survive is how you initialize the weights.

---

## Saturation: when neurons stop listening

Take the sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

and its derivative:

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

Now imagine the input to sigmoid becomes very large or very small.  
For $x = 10$, the output is almost $1$.  
For $x = -10$, the output is almost $0$.

In both cases, the derivative is almost zero.

That’s where trouble begins.

During backpropagation, gradients keep getting multiplied by this tiny number. After a few layers, they’re basically gone. The neuron still produces output, but it has almost no ability to change anymore.

This is saturation: not dead, but stuck.

If this happens to most neurons in a layer, your deep network suddenly behaves like a very small one. Training becomes slow, sometimes impossible. Good initialization avoids this by keeping activations away from the extremes of the curve, right where gradients are strong and learning actually happens.

---

## Why all-zero weights kill learning

Now imagine you initialize everything to zero. Two neurons in the same layer look like this:

$$
\text{neuron}_1 = w_{11}x_1 + w_{12}x_2 = 0
$$

$$
\text{neuron}_2 = w_{21}x_1 + w_{22}x_2 = 0
$$

They produce the same output. They receive the same gradients. They make the same updates.

After one step:

$$
w_{11} = w_{21}, \qquad w_{12} = w_{22}
$$

And they stay identical forever.

Your network might have 100 neurons per layer, but it behaves like it has only one. That’s why weights must start random — to break symmetry so neurons can learn different things.

But randomness alone isn’t enough. The *scale* of that randomness matters even more.

---

## Too small, too big — both are disasters

If weights are too small, activations shrink layer by layer:

$$
0.5 \rightarrow 0.1 \rightarrow 0.01 \rightarrow \dots \rightarrow 0
$$

Everything fades out. Gradients vanish. Learning stops.

If weights are too large, activations blow up:

$$
2 \rightarrow 100 \rightarrow 10{,}000 \rightarrow \dots
$$

Sigmoid and tanh saturate immediately. Gradients die again. Different reason, same result.

So the real goal of initialization is simple:

> Keep the size of activations roughly the same as they move through layers.

Not growing.  
Not shrinking.  
Stable.

---

## The math behind stable signals

Every neuron computes the same kind of sum:

$$
s = w_1x_1 + w_2x_2 + \dots + w_nx_n
$$

If the inputs have variance $\mathrm{Var}(x)$ and the weights have variance $\mathrm{Var}(w)$, then the output variance is

$$
\mathrm{Var}(s) = n \times \mathrm{Var}(w) \times \mathrm{Var}(x)
$$

Stack layers and this effect compounds:

$$
\mathrm{Var}(s_k) = [n \times \mathrm{Var}(w)]^k \times \mathrm{Var}(x)
$$

This single equation explains almost every initialization failure.

If

$$
n \times \mathrm{Var}(w) > 1
$$

activations explode.

If

$$
n \times \mathrm{Var}(w) < 1
$$

activations vanish.

So the sweet spot is boring and precise:

$$
n \times \mathrm{Var}(w) = 1
$$

That’s it. Keep the signal the same size across depth.

---

## Why everything comes back to $(1/\sqrt{n})$

A neuron adds up $n$ random numbers. When you add random numbers, the total doesn’t grow like $n$. It grows like $\sqrt{n}$.

So naturally,

$$
s \approx \sqrt{n}
$$

If we want the output to stay the same size as the input, weights must cancel this growth:

$$
\text{weight size} \times \sqrt{n} \approx 1
$$

which gives

$$
\text{weight size} \approx \frac{1}{\sqrt{n}}
$$

Not $1$ — that explodes.  
Not $1/n$ — that kills everything.  
$\frac{1}{\sqrt{n}}$ is the balance point.

Every modern initialization method is basically this idea wearing different clothes.

---

## Xavier initialization

For sigmoid, tanh, and linear layers, Xavier keeps things stable by following the variance rule:

$$
W \sim \mathcal{N}\!\left(0, \frac{1}{\text{$fan\_in$}}\right)
$$

This gives just enough randomness at exactly the right scale so signals neither blow up nor fade away.

---

## ReLU changed the game

ReLU is simple:

$$
\text{ReLU}(x) = \max(0, x)
$$

Half the values become zero. Always.

So even if Xavier keeps things mathematically balanced, ReLU quietly cuts your signal in half at every layer. After a few layers, your network becomes a whisper.

That’s why Xavier isn’t enough for ReLU networks.

---

## He initialization: made for ReLU

He initialization fixes this by starting slightly bigger:

$$
W \sim \mathcal{N}\!\left(0, \frac{2}{\text{$fan\_in$}}\right)
$$

That factor of $2$ is not decoration.  
It’s compensation for the half of the signal that ReLU throws away.

With this scaling, activations stay healthy even in very deep networks. Gradients flow. Learning works.

That’s why almost every modern ReLU network relies on He initialization, even if you never think about it.

---

## What you should actually use

Most of the time, the rule is simple:

- Using **ReLU or its variants** → He initialization  
- Using **sigmoid or tanh** → Xavier initialization  
- Using **pretrained models** → keep their weights, only initialize new layers  

Frameworks already do this for you. But knowing *why* these defaults exist makes debugging a lot easier when things go wrong.

---


