---
layout: post
title: Bias-Variance Tradeoff and Need of Regularization 
date: 2026-01-02 17:18 +0530
math: true
description : This blog gives you the understanding of What is bias , what is variance , and how training a model is doing a tradeoff between bias and variance.
---
---



When we begin learning machine learning, terms like *overfitting*, *bias–variance tradeoff*, and *regularization* often appear together. At first, they feel abstract and mathematical. But at their core, all of them try to answer one fundamental question:

**How well will my model perform on unseen data?**

In this blog, we’ll build this understanding step by step, starting from simple intuition and slowly connecting it to the math behind it.

---

## Simple Models vs Complex Models: An Intuitive Experiment

Imagine you have a fixed dataset and two kinds of models:

* a **simple model**, such as linear regression
* a **complex model**, such as a high-degree polynomial or a deep neural network

Now suppose you repeatedly sample different subsets from the full dataset and train multiple models of the same type on those subsets.

What do you observe?

The simple model usually performs poorly. It fails to capture the underlying structure of the data. However, its predictions across different samples look quite similar.

The complex model performs much better on average. But when trained on different subsets, its predictions differ significantly.

This behavior naturally leads us to the concepts of **bias** and **variance**.

---

## What Is Bias?

Bias measures how far the *average prediction* of a model is from the true function.

If $$ ( \hat f(x) ) $$ is the learned model and $$ ( f(x) ) $$ is the true underlying function, bias is defined as:

$$
\text{Bias}(\hat f(x)) = \mathbb{E}[\hat f(x)] - f(x)
$$

Simple models make strong assumptions about the data. Because of these assumptions, they are unable to capture complex patterns. As a result, their predictions systematically deviate from the true function.

This is why simple models are said to have **high bias**.

---

## What Is Variance?

Variance measures how much the model’s predictions change when it is trained on different datasets.

Formally:

$$
\text{Variance}(\hat f(x)) =
\mathbb{E}\big[(\hat f(x) - \mathbb{E}[\hat f(x)])^2\big]
$$

Complex models are highly flexible. Even small changes in the training data can lead to large changes in the learned function. This sensitivity makes their predictions unstable.

This is why complex models are said to have **high variance**.

---

## The Bias–Variance Tradeoff

From these definitions, a clear pattern emerges:

* Simple models → **high bias, low variance**
* Complex models → **low bias, high variance**

Reducing bias usually increases variance, and reducing variance usually increases bias. This unavoidable tension is called the **bias–variance tradeoff**.

---

## How Bias and Variance Affect Prediction Error

The expected prediction error can be decomposed as:

$$
\mathbb{E}\big[(y - \hat f(x))^2\big]
=
\text{Bias}^2
+
\text{Variance}
+
\sigma^2
$$

Here:

* Bias² represents error due to underfitting  
* Variance represents error due to overfitting  
* $$ ( \sigma^2 ) $$ is irreducible noise in the data  

No model can eliminate the noise term. Our goal is to balance bias and variance.

---

## Training Error vs Test Error

In practice, we evaluate models using training and test (or validation) error.

For a dataset with \( n \) training points and \( m \) validation points:

**Training error** is defined as:

$$
\text{train error}
=
\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat f(x_i))^2
$$

**Test error** is defined as:

$$
\text{test error}
=
\frac{1}{m}\sum_{i=n+1}^{n+m}(y_i - \hat f(x_i))^2
$$

As model complexity increases, training error always decreases. A sufficiently complex model can fit the training data almost perfectly.

Test error behaves differently. Initially, it decreases as the model learns useful patterns. Beyond a certain point, it starts increasing as the model begins to fit noise rather than signal. This creates the familiar U-shaped curve, with a **sweet spot** where generalization is best.

---

## Why Training Error Is Overly Optimistic

To understand this, assume the data is generated as:

$$
y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

The quantity we truly care about is the **true error**:

$$
\mathbb{E}\big[(\hat f(x) - f(x))^2\big]
$$

But since the true function \( f(x) \) is unknown, we cannot compute this directly. Instead, we compute the empirical error:

$$
\mathbb{E}\big[(\hat y - y)^2\big]
$$

Expanding this expression gives:

$$
\text{True Error}
=
\text{Empirical Error}
-
2\,\mathbb{E}[\epsilon(\hat f(x) - f(x))]
+
\sigma^2
$$

This additional term explains everything.

For **test data**, the noise is independent of the model because the test set was not used during training. The covariance term vanishes, making test error a good approximation of true error.

For **training data**, the noise influences the learned parameters. This makes the covariance term positive, which causes training error to underestimate true error.

This is why training error gives an overly optimistic picture.

---

## Model Complexity and Variance

Using Stein’s lemma, the extra error term can be written as:

$$
\sigma^2 \cdot \frac{1}{n}
\sum_{i=1}^{n}
\frac{\partial \hat f(x_i)}{\partial y_i}
$$

This quantity measures how sensitive the model is to individual data points.

Simple models change very little when a data point changes. Complex models change a lot. This sensitivity is exactly what we call **variance**.

So we can think of true error as:

$$
\text{True error}
=
\text{Training error}
+
\text{Model complexity penalty}
$$


![Bias vs Variance Tradeoff](https://cdn.analyticsvidhya.com/wp-content/uploads/2024/07/eba93f5a75070f0fbb9d86bec8a009e9.webp)
---

## Why Regularization Is Necessary

If we minimize only the training error, the optimizer naturally prefers highly flexible models. This reduces bias but increases variance, leading to overfitting.

Regularization fixes this by modifying the objective:

$$
\min_\theta
\Big(
L_{\text{train}}(\theta)
+
\lambda\,\Omega(\theta)
\Big)
$$

The penalty term discourages excessive complexity.

---

## Regularization as Variance Control

Different regularization techniques achieve this in different ways:

* L2 regularization penalizes large weights  
* L1 regularization encourages sparse solutions  
* Early stopping prevents fitting noise  
* Dropout injects randomness in hidden layers  
* Data augmentation enforces invariances  
* Input noise smooths the learned function  

All of them reduce model sensitivity and, therefore, variance.

---

## Final Thoughts

The goal of machine learning is **not** to minimize training error.

The real goal is to minimize **true error** — the error on unseen data.

The bias–variance tradeoff explains why this is difficult, and **regularization** is the tool that allows us to strike the right balance.

---

**Next up:** we’ll dive deeper into different regularization techniques and understand how each one controls model complexity in practice.

---
