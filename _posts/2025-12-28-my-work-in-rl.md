---
layout: post
title: My Journey Through Reinforcement Learning 
date: 2025-12-28 13:50 +0530
math: true
description : This is a walk through my incomplete but amazing journey learning RL.
---

## Introduction

Reinforcement Learning (RL) has always fascinated me because of its simplicity at the core and its power in practice.  
At its heart, RL is about learning **through interaction** — an agent observes the environment, takes actions, receives rewards, and gradually improves its behavior.

Over time, I explored RL from multiple angles:
- **Tabular methods** (Q-learning from scratch)
- **Deep value-based methods** (DQN and its variants)
- **Policy-based methods** (REINFORCE)

This post is a walkthrough of my Reinforcement Learning projects, the ideas behind them, and what I learned while building everything largely **from scratch**.

---

## 1. Q-Learning From Scratch

📌 **Repository:**  
👉 https://github.com/YashrajKupekar17/Q-learning

### Overview

Q-learning is one of the most fundamental RL algorithms.  
It is **model-free**, **off-policy**, and learns an optimal action-value function using the Bellman equation:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s, a) \right]
$$

This project implements Q-learning **from scratch**, without relying on high-level RL libraries.

### Environments

I tested the implementation on multiple environments from OpenAI Gym:

- **FrozenLake**
  - 4×4 (Non-slippery)
  - 8×8 (Slippery)
- **Custom Taxi Environment**
  - 500 states
  - 7200 states

These environments helped highlight:
- exploration vs exploitation
- the impact of stochastic transitions
- scalability issues of tabular methods

### Interactive Notebooks

I provide Colab notebooks for hands-on experimentation:
- Q-learning implementation
- FrozenLake (8×8, slippery)
- Custom Taxi environment

### Trained Models

To make evaluation easier, trained Q-tables are hosted on Hugging Face:

- FrozenLake (4×4, Non-slippery)
- FrozenLake (8×8, Slippery)
- Taxi (500 states)
- Taxi (7200 states)

This project solidified my understanding of **Bellman updates, convergence, and exploration strategies**.

---

## 2. Deep Q-Learning on Atari Breakout

📌 **Repository:**  
👉 https://github.com/YashrajKupekar17/Breakout

### Why Deep Q-Networks?

Tabular Q-learning breaks down when the state space becomes large.  
For example, a raw Atari frame (210×160×3 pixels) makes a Q-table impossible.

Deep Q-Networks (DQN) solve this by using a **neural network** to approximate the Q-function directly from pixels.

### Environment & Setup

- **Environment:** ALE/Breakout-v5
- **Policy:** `CnnPolicy`
- **Frame stacking:** 4 frames
- **Training:** 1M timesteps
- **Library:** `rl-baselines3-zoo`

Key techniques used:
- Experience replay
- Target networks
- ε-greedy exploration
- CNN-based value approximation

This project helped me understand how **deep learning stabilizes RL** and why architectural choices matter when learning directly from images.

---

## 3. DQN, Double DQN, and Dueling DQN (From Scratch)

📌 **Repository:**  
👉 https://github.com/YashrajKupekar17/DQN-pytorch-Cartpole-Flappy_bird

### Overview

In this project, I implemented multiple DQN variants **from scratch in PyTorch**:

- **Vanilla DQN**
- **Double DQN** (reduces overestimation bias)
- **Dueling DQN** (separates value and advantage)

### Environments

- **CartPole-v1**
- **FlappyBird-v0**

### Key Features

- Modular architecture for switching between DQN variants
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration with decay
- YAML-based hyperparameter configuration
- Training visualization and video recording

### Results

- **CartPole-v1:** Solved (475+ average reward)
- **FlappyBird-v0:** Agent learns to navigate pipes effectively  
  (training limited due to compute constraints)

This project taught me:
- why Double DQN improves stability
- how Dueling DQN helps in states where action choice matters less
- how implementation details strongly affect learning dynamics

---

## 4. Policy Gradient (REINFORCE)

📌 **Repository:**  
👉 https://github.com/YashrajKupekar17/Policy-Gradient-Pytorch_Implementation-Pixelcopter

### Why Policy Gradients?

Unlike value-based methods, **policy gradient algorithms directly optimize the policy**.  
REINFORCE updates parameters by maximizing expected return:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a|s) \, G_t \right]
$$

### Environments

- **CartPole-v1**
- **Pixelcopter-v0** (Box2D, pixel-based)

### Highlights

- Softmax policy
- Reward-based learning without Q-values
- Works on both low-dimensional and pixel-based environments
- Training visualizations and GIFs included

This project clarified:
- the high variance nature of policy gradients
- the importance of baselines and normalization
- how policy-based methods differ fundamentally from value-based RL

---

## Key Takeaways

Across these projects, I gained hands-on experience with:

- Tabular vs Deep RL trade-offs
- Exploration strategies and stability issues
- Value-based vs policy-based learning
- Architectural improvements in DQN
- Practical challenges like compute limits and hyperparameter tuning

---

## Future Directions

Some improvements I plan to explore next:

- Prioritized Experience Replay
- Noisy Networks for exploration
- Rainbow DQN
- Actor–Critic methods (A2C, PPO)
- Better hyperparameter optimization

---

## Closing Thoughts

Reinforcement Learning is challenging, unstable, and incredibly rewarding to work with.  
Building these algorithms from scratch gave me intuition that no high-level library alone could provide.

If you’re learning RL, I highly recommend implementing things yourself at least once — the insights are worth it.
