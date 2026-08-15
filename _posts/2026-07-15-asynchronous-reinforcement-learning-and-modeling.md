---
layout: post
title: "How to Model Asynchronous Multi-Agent Systems and Design Learning Algorithms?"
date: 2026-07-15
description: "In reality, most multi-agent systems operate asynchronously, such as cloud scheduling, traffic networks, and wireless spectrum markets, while existing multi-agent reinforcement learning methods generally assume simultaneous decision-making by all agents and model these systems as Markov games. This article will answer how asynchronous multi-agent systems can be modeled directly without delay-based approximations and how reinforcement learning algorithms can be designed for them."
tags:
  - Reinforcement Learning
  - Multi-Agent Learning
thumbnail: assets/img/Asynchronous-RL/02-temporal-mean-field-framework.png
related_posts: false
related_publications: true
images:
  slider: true
---

# Asynchronous Multi-Agent System

Multi-agent reinforcement learning (MARL) is generally built on the basic assumption that agents make decisions simultaneously. Nash Q-learning, for example, requires all agents to jointly determine a Nash equilibrium in every stage game. Policy-based MARL adopts the same synchronous Markov-game formulation, where state transitions and learning updates are based on the joint action produced by all agents acting simultaneously.

However, in reality, decisions in most complex systems are made asynchronously.
1. **Large-scale computing systems**: Jobs arrive and computing resources become available at different times, so scheduling and resource-allocation decisions are made in response to different events.
2. **Transportation systems**: Vehicle and passenger arrivals, movements, and departures occur continuously, and each agent makes mobility decisions at its own decision time.
3. **Communication systems**: Devices initiate transmissions at different times according to their local data arrivals, resulting in asynchronous channel-access and resource-allocation decisions.

{% include figure.liquid path="assets/img/Asynchronous-RL/01-asynchronous-multi-agent-system.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Asynchronous multi-agent reinforcement learning system" %}

Existing methods mainly accommodate asynchronous decision-making by converting it into a synchronous decision process. For example, delay-based methods map different action-completion times to delays on a shared time grid; macro-action methods encapsulate actions of different durations while retaining synchronized primitive-action transitions; and padding-based methods assign no-op or repeated actions to inactive agents so that a complete joint action is available at every step. 

> However, these treatments are compromises made to retain the synchronous underlying model. Even when agents make decisions asynchronously, these methods must still convert their interactions into synchronized joint transitions. Without this conversion, the validity of the learning updates and the corresponding convergence guarantees cannot be established.

In the following sections, we first take the mean-field setting as an example to introduce a new asynchronous framework and identify the conditions that guarantee both the existence and uniqueness of a Nash policy and the convergence of policy-gradient-based algorithms.

# Temporal Mean Field (TMF) Framework

We first consider a simplified system in which a fixed number $B_t$ of agents make decisions at each time step $t$.

## Modeling

Considering that the observation space $\mathcal O$ is finite and agents are homogeneous and exchangeable, we can directly represent the system dynamics using the population distribution $\mu_t$, where $\mu_t(o)$ denotes the fraction of the $N$ agents whose observation is $o$ at time step $t$. The population distribution evolves through the transitions of active and passive agents:

$$
\bar{\mu}_{t+1}(o)
=
\sum_{o'\in\mathcal{O}}\mu_t(o')\left[
\underbrace{
\frac{B_t}{N}
\sum_{a\in\mathcal{A}}
\pi(a\mid o',\mu_t)
P(o\mid o',a,\mu_t)
}_{\text{active-agent transitions}}
+
\underbrace{
\frac{N-B_t}{N}
P_0(o\mid o',\mu_t)
}_{\text{passive-agent transitions}}
\right].
$$

Here, $\pi$ is the shared policy, while $P$ and $P_0$ are the transition functions for active and passive agents, respectively. The resulting $\bar{\mu}_{t+1}$ is the population distribution at the next time step.

{% include figure.liquid path="assets/img/Asynchronous-RL/02-temporal-mean-field-framework.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Temporal mean field modeling framework" %}

Given the population dynamics, we can define the expected return of an active agent under the shared policy $\pi$ as

$$
V^\pi(o',\mu_t)=\sum_{a\in\mathcal A}\pi(a\mid o',\mu_t)\left[r(o',a,\mu_t)+\gamma\sum_{o\in\mathcal O}P(o\mid o',a,\mu_t)V^\pi(o,\mu_{t+1})\right].
$$

Based on this value function, we can define a new equilibrium that couples the shared policy with the population trajectory. Let $\{\mu_t^{\ast}\}$ be the population trajectory generated by $\pi^{\ast}$ through the population dynamics above. The policy $\pi^{\ast}$ is an equilibrium policy if

$$
V^{\pi^{\ast}}(o',\mu_t^{\ast})\geq V^\pi(o',\mu_t^{\ast})\quad \forall\,\pi,\,o',\,t.
$$

> TMF is applicable when agents are `homogeneous`, `exchangeable` and their interactions exhibit some form of `congestion effect`. Representative examples include anonymous resource-selection games, parallel-server load balancing, routing among homogeneous vehicles, and large-scale queueing systems. 

> The `existence` and `uniqueness` of the TMF equilibrium are guaranteed by Assumption 4.3, discrete monotonicity, which is more general than assuming an explicit congestion structure. Under an additional condition linking the interaction coupling and batch size, the finite-agent system admits a `bounded approximation error` relative to its mean-field counterpart.

## Learning Algorithm

The TMF Policy Gradient (TMF-PG) algorithm is illustrated below. It allows agents to make decisions asynchronously, with the joint action replaced by the population distribution $\hat{\mu}_t$. We prove that `TMF-PG converges to the unique TMF equilibrium` established above when sufficient policy-gradient updates are performed at each iteration.

{% include figure.liquid path="assets/img/Asynchronous-RL/03-tmf-reinforcement-learning.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="100%" zoomable=true alt="TMF reinforcement learning algorithm" %}




---
{% nocite yang_mean_field %}
