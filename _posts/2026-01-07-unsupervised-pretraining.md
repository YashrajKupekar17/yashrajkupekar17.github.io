---
layout: post
title: Unsupervised Pretraining -> How We Learned to Train Deep Networks
date: 2026-01-07 19:07 +0530
math: true
description: What is Unsupervised pretraining and how it opened doors for different researches in Deep Learning ?
---

Training a neural network is, at its core, a game of gradients.

Check this blog for Backpropagation : [Backpropagation](https://yashrajkupekar17.github.io/posts/backpropagation/)

Backpropagation was already known in the 1980s and 1990s. Yet for a long time it simply did not work well for deep networks. As depth increased, training stalled. Loss refused to go down. Gradients vanished before they reached the early layers.

So the natural question was obvious:

**If deep networks were so hard to train, how did deep learning suddenly become successful?**

The answer is surprisingly recent.

Until the mid-2000s, deep neural networks were largely seen as impractical. That changed after a single influential paper:

**Hinton & Salakhutdinov (2006), *Reducing the dimensionality of data with neural networks*.**

This work introduced a simple idea that changed the direction of the field: **unsupervised pretraining**.

---

## Train before you train

Instead of training a deep network end-to-end from random initialization, Hinton proposed a two-stage process.

First, train the network in an unsupervised way, layer by layer, so that it learns meaningful representations of the data.  
Then, fine-tune the entire network using labeled data.

The key idea was that deep networks failed not because they were incapable of representing functions, but because they were extremely hard to train from scratch.

---

## Layer-wise unsupervised pretraining

Consider a deep neural network with multiple hidden layers. Instead of training it all at once, we train it greedily.

We start with the first layer and train a simple autoencoder:

$$
x \rightarrow h_1 \rightarrow \hat{x}
$$

The objective is purely reconstruction:

$$
\min \sum (x - \hat{x})^2
$$

There are no labels involved. The network is only trying to learn a representation $( h_1 )$ that captures the structure of the input data.

Once this layer is trained, we freeze it.

<figure style="text-align: center;">
  <img src="/images/Unsupervised_pretraining.png"
       alt="Unsupervised Pretraining"
       width="400">
  <figcaption style="font-size: 0.9em; color: #666;">
    Diagram from Prof. Mitesh Khapra's Lecture
  </figcaption>
</figure>

Now we treat its output $ ( h_1 ) $ as input and train another autoencoder on top:

$$
h_1 \rightarrow h_2 \rightarrow \hat{h}_1
$$

Again, the goal is reconstruction, this time of the previous layer’s activations.

We repeat this process layer by layer. Each layer learns a slightly more abstract representation of the data. By the end, every layer has learned something meaningful before seeing a single label.

Finally, we attach the output layer and fine-tune the entire network using supervised learning:

$$
\min_\theta \frac{1}{m} \sum_{i=1}^{m} (y_i - f(x_i))^2
$$

Now we are no longer starting from random weights. We are starting from a network that already understands the structure of the data.

---

## Why did this help?

Empirically, networks trained this way converged faster and generalized better than networks trained from scratch. The big question was why.

Explanations that stood out: optimization and regularization.

---

### Optimization

Training deep networks means minimizing a highly non-convex loss surface:

$$
L(\theta) = \frac{1}{m} \sum (y_i - f(x_i))^2
$$

With random initialization, gradient descent often starts in bad regions of this landscape, especially when the network is deep. Gradients can vanish, learning can stall, and optimization may never reach a good solution.

Unsupervised pretraining changes the starting point. It places the parameters in a region where the network already represents useful structure in the data. Fine-tuning then starts close to a good basin of attraction instead of wandering blindly.

Later work showed that if a network has very high capacity, it can often fit the training data even without pretraining. But when capacity is limited, which was common in early deep learning, pretraining made optimization much easier.

---

### Regularization

Pretraining also acts as a form of regularization.

During pretraining, the network minimizes a reconstruction objective:

$$
\Omega(\theta) = \sum (x - \hat{x})^2
$$

This forces the parameters into regions of the space that preserve information and capture real structure in the data. When we later minimize the supervised loss,

$$
L(\theta) = \sum (y - f(x))^2
$$

we are no longer free to choose any solution that fits the labels. We are constrained to solutions that also make sense for the input distribution.

In effect, the model is learning labels using representations that were already shaped by the data itself. That is regularization, even though it does not look like L2 penalties or dropout.

---

## The real impact of pretraining

In hindsight, the most important contribution of unsupervised pretraining was not just that it worked.

It was that researchers did not fully understand why it worked.

Before pretraining, the common belief was that deep networks failed because they were too complex. After pretraining, the belief shifted toward a different explanation: deep networks fail because they are hard to train.

That shift changed the kinds of questions people asked. Was the problem optimization? Initialization? Activation functions? Gradient flow? Training instability?

Once these questions became clear, researchers started addressing them directly.

---

## From one trick to many ideas

Instead of asking how to pretrain networks, the field began asking how to make pretraining unnecessary.

That line of thinking led to many of the techniques we now take for granted:

* Better activation functions like ReLU to improve gradient flow  
* Smarter initialization methods like Xavier and He  
* Normalization techniques like BatchNorm  
* More stable optimizers like RMSProp and Adam  

In this sense, unsupervised pretraining was not the final solution. It was the first serious diagnosis of what was wrong with deep learning.

---

## Why we rarely use it today

Unsupervised pretraining is rarely used in classical deep learning pipelines today, not because it was incorrect, but because newer methods solve the same problems more directly.

Modern activations, initialization schemes, normalization, and optimizers largely remove the training pathologies that made pretraining necessary in the first place.

Still, historically, pretraining was the bridge that took the field from “deep networks don’t work” to “deep learning changes everything.”

---

In the next blog , I will discuss about better activation functions .

---

## References

* Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the dimensionality of data with neural networks*. Science.  
* Rumelhart, Hinton, Williams — Backpropagation
  <https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf>  
* Larochelle et al. (2009). *Exploring Strategies for Training Deep Neural Networks*  
  <https://www.jmlr.org/papers/v10/larochelle09a.html>  
* Erhan et al. (2009). *The Difficulty of Training Deep Architectures and the Effect of Unsupervised Pre-Training*  
  <https://proceedings.mlr.press/v5/erhan09a/erhan09a.pdf>  
* Erhan et al. (2010). *Why Does Unsupervised Pre-training Help Deep Learning?*  
  <https://jmlr.org/papers/volume11/erhan10a/erhan10a.pdf>  
* *Unsupervised pretraining code example* <https://rpubs.com/kev22nov/greedylayerwiseunsupervisedtrainingprotocol>
* *Prof. Mitesh Khapra's CS6910 course* <https://www.cse.iitm.ac.in/~miteshk/CS6910.html>