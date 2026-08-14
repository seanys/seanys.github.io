---
layout: post
title: "Why Do We Need Analytical Models to Guide Training for Scalable Multi-Agent Learning?"
date: 2026-08-14
description: Using analytical models to provide structured guidance for reinforcement learning.
tags:
  - Reinforcement Learning
  - Model-Augmented Learning
thumbnail: assets/img/DG-PG/01-cloud-task-dispatching.jpeg
related_posts: false
related_publications: true
images:
  slider: true
---

# Background
Cooperative multi-agent reinforcement learning (CMARL) refers to settings in which agents share a common objective and must coordinate their actions, such as cloud scheduling and power-system operation. The training difficulty of cooperative MARL grows rapidly with the number of agents $N$, stemming from multi-agent credit assignment: when all agents jointly contribute to the shared objective, no individual agent's contribution can be cleanly identified. 

Taking QMIX as an example, all agents are trained using the same team reward, while the learning algorithm must identify each agent's individual contribution from this shared signal, causing the noise to scale as $\Theta(N)$.

{% include figure.liquid path="assets/img/DG-PG/02-marl-gradient-variance.jpeg" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Policy-gradient variance in multi-agent reinforcement learning" %}

As the number of agents increases, the $\Theta(N)$-scaling noise increasingly dominates the useful gradient signal, making reliable policy updates progressively more difficult. Model-based counterfactual methods, for example, rely on accurately modeling how alternative individual actions affect system outcomes, which requires an accurate dynamics model that may be unavailable in practice and costly to obtain.

# Motivation and Proposed Approach
**Fortunately, many cooperative engineering systems admit differentiable analytical models that prescribe efficient system states. For example, queueing models characterize how workloads should be balanced across servers in cloud scheduling, while power-flow equations characterize how generation should meet demand subject to network constraints in power systems. These prescriptions remain system-level descriptions and do not directly yield local decision rules for individual agents.**
<div class="w-75 mx-auto">
    <swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
        <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/DG-PG/03-multi-server-queueing-model.jpeg" class="img-fluid rounded z-depth-1" zoomable=true alt="Analytical model of a load-balanced multi-server queueing system" %}</swiper-slide>
        <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/DG-PG/04-base-stock-policy.jpeg" class="img-fluid rounded z-depth-1" zoomable=true alt="Base-stock policy as an analytical model" %}</swiper-slide>
    </swiper-container>
</div>
Departing from existing approaches, we propose a new direction in which we directly use analytical models to obtain an optimal system state $\tilde{\boldsymbol{x}}=\left\lbrace\tilde{x}^n\right\rbrace_{n=1}^N$ and use this state to guide policy training. Examples include the load-balancing state in cloud scheduling, the economic-dispatch solution in power systems, and the system-optimal flow pattern in transportation networks. We only need to guide the system toward this reference state to achieve near-optimal system performance.

For example, in cloud scheduling, the optimal system state corresponds to load balance; in power systems, to optimal power flow; and in transportation networks, to system-optimal traffic flow. We only need to guide policy training so that the learned policies can more easily drive the system toward the corresponding optimal state.

Within this system representation, each component $x_t^n$, such as the workload of a server, the power output of a generator, or the flow on a road link, is generally affected by only a limited subset of agents. This local dependence allows us to construct a low-variance guidance gradient for agent $i$ as

$$
\hat{g}_{\mathcal{G},t}^i
=
\left\langle
\boldsymbol{x}_t-\tilde{\boldsymbol{x}}_t,
\boldsymbol{z}_t^i
\right\rangle
\nabla_{\theta^i}\log\pi^i(a_t^i\mid o_t^i),
$$

where $\boldsymbol{z}_t^i\triangleq\frac{\partial\boldsymbol{x}_t}{\partial a_t^i}$ represents the local influence of agent $i$ on the system state.


{% include figure.liquid path="assets/img/DG-PG/07-gradient-variance-reduction.jpeg" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="" %}

Taking cloud scheduling as an example, $\boldsymbol{x}_t-\tilde{\boldsymbol{x}}_t$ measures how the current workload of each server deviates from the load-balanced state, while $\boldsymbol{z}_t^i\triangleq\frac{\partial\boldsymbol{x}_t}{\partial a_t^i}$ directly indicates how agent $i$'s action changes the workload of each server. **Consequently, $\hat{g}_{\mathcal G,t}^i$ identifies whether the action moves the system toward or away from load balance, providing a low-variance guidance signal for policy training.**

{% include figure.liquid path="assets/img/DG-PG/06-deviation-and-local-influence.jpeg" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Deviation from the analytical reference and local action influence" %}

# Implementation

In implementation, DG-PG can be incorporated into any policy-gradient-based algorithm by adding one term to its original advantage. For example, in PPO, we just need to replace the original PPO advantage $\hat A_{\mathrm{PPO},t}^i$ with

$$
\hat A_{\mathrm{DG},t}^i
=
(1-\alpha)\hat A_{\mathrm{PPO},t}^i
-
\alpha g_{\mathcal G,t}^i,
$$

where $\alpha\in[0,1]$ controls the balance between the original PPO advantage and the analytical guidance. 


Computing $g_{\mathcal G,t}^i$ is also straightforward. Taking cloud scheduling as an example, suppose agent $i$ assigns a task to server $a_t^i$, where each task requires both CPU and memory. After computing the target CPU and memory utilizations $\tilde{x}\_{\mathrm{CPU},t}^n$ and $\tilde{x}\_{\mathrm{MEM},t}^n$ for every server $n$, the guidance term is

$$
g_{\mathcal G,t}^i
=
\sum_{n=1}^{N}
\left[
\left(x_{\mathrm{CPU},t}^n-\tilde{x}_{\mathrm{CPU},t}^n\right)
z_{\mathrm{CPU},t}^{i,n}
+
\left(x_{\mathrm{MEM},t}^n-\tilde{x}_{\mathrm{MEM},t}^n\right)
z_{\mathrm{MEM},t}^{i,n}
\right],
$$

where $z_{\mathrm{CPU},t}^{i,n}$ and $z_{\mathrm{MEM},t}^{i,n}$ directly equal the CPU and memory demands associated with agent $i$'s action on server $n$, respectively. The resulting advantage is passed directly to the original PPO update. The critic, reward function, and all other components remain unchanged.

```python
# analytical_state(action) returns the system state induced by this action
x = analytical_state(action)
local_influence = torch.autograd.functional.jacobian(analytical_state, action)
dgf = torch.dot(x - x_opt, local_influence)
advantage = (1 - alpha) * advantage - alpha * dgf
```

When the action is discrete, the local influence is directly given by the change in $\boldsymbol{x}$ caused by the selected action.

For implementation details, see the repository below.

<div class="repositories d-flex justify-content-center">
  {% include repository/repo.liquid repository="seanys/Descent-Guided-Policy-Gradient-for-Scalable-Cooperative-Multi-Agent-Learning" %}
</div>

--- 
{% nocite yang_descent_guided %}
