---
layout: post
title: "How to Routing for Connected Vehicles?"
date: 2026-04-15
description: ""
tags:
  - Reinforcement Learning
  - Transportation
thumbnail: assets/img/MRG/connected-vehicles.jpg
related_posts: false
related_publications: true
images:
  slider: true
---

Traffic events such as accidents, roadworks, and adverse weather make travel times in road networks inherently stochastic. These events are also major sources of congestion. Delayed route adjustment causes incident-induced queues to propagate upstream and spill back into adjacent links. While the concentration of diverted traffic on a single alternative route may transfer congestion across the network, giving rise to the flash crowd effect.

With advances in communication technologies, connected vehicles (CVs) can access real-time traffic information and exchange their intended route choices with other vehicles. These capabilities enable real-time joint adaptive routing in stochastic road networks, accommodating individual routing preferences while mitigating congestion caused by collective overreaction. 

Coordinating the routing decisions of all CVs, however, remains challenging. Each vehicle must simultaneously account for the actions of all other vehicles, whose destinations and routing preferences differ. Moreover, since vehicles affect one another whenever their routes overlap, these decisions are also coupled through the road-network topology.

{% include figure.liquid path="assets/img/MRG/cv-routing-coordination.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Coordinated routing decisions for connected vehicles" %}

We therefore begin by proposing a Markov Routing Game (MRG), as illustrated below. In the single-agent setting, the routing problem is generally modeled as a Markov decision process (MDP), with intersections defined as decision nodes and outgoing links as actions. In the multi-agent setting, however, directly combining these single-agent MDPs results in asynchronous decisions across vehicles, making it difficult to guarantee a joint policy under which each agent responds optimally to the others. To address this issue, we formulate the routing decision process differently in the MRG.

{% include figure.liquid path="assets/img/MRG/learning-process.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Routing-policy learning process" %}

In our MRG, all vehicles make decisions simultaneously every $\Delta t$, independently of whether they are approaching an intersection. An action represents the vehicle’s expected trajectory during the next $\Delta t$. For instance, $a_t^n=[0,0.4,0,0.6,\ldots]$ means that vehicle $n$ is expected to spend 40% of this period on the 2nd link and 60% on the 4th link, with all entries of the action vector summing to one. The action space remains enumerable and can be expressed using the road-network graph, although the feasible trajectory choices depend on the vehicle’s current observation. 

The figure below presents an example of a vehicle observation, where $[0.5,8,1.2,1]$ represents its relative position, speed, acceleration, and vehicle type, respectively. 

{% include figure.liquid path="assets/img/MRG/mean-observation.png" class="img-fluid rounded z-depth-1 d-block mx-auto" zoomable=true  width="50%"  alt="Mean-observation representation for connected-vehicle routing" %}

To achieve scalable learning in complex environments, we introduce the homogeneity-based mean-field approximation (HMFA), which separately approximates the joint action and joint observation. For the joint action, vehicles with homogeneous routing behavior are grouped together, and their individual actions are aggregated into a mean action for each group. Accordingly, the mean action of each group represents the expected average number of vehicles from that group occupying each link over the next $\Delta t$. In the figure below, for example, the value 15 indicates an average of 15 vehicles from Group 1 on the corresponding link.

{% include figure.liquid path="assets/img/MRG/group-mean-field.png" class="img-fluid rounded z-depth-1 d-block mx-auto" zoomable=true width="50%"  alt="Group-based mean-field representation" %}

The learning framework is illustrated below. To ensure convergence of the learning algorithm, at each decision epoch, agents repeatedly exchange their intended actions and update their policies based on the group mean actions from the previous iteration:

$$
\pi^n(\cdot\mid s,\boldsymbol{o})
=
\frac{
\exp\!\left(
\beta Q_t^n
\left(
s,o^n,\bar{o}^n,a^n,
\bar{a}_{-}^1,\ldots,\bar{a}_{-}^C
\right)
\right)
}{
\displaystyle
\sum_{a^n\in\mathcal A^n}
\exp\!\left(
\beta Q_t^n
\left(
s,o^n,\bar{o}^n,a^n,
\bar{a}_{-}^1,\ldots,\bar{a}_{-}^C
\right)
\right)
},
$$

$$
a^n\sim\pi^n(\cdot\mid s,\boldsymbol{o}).
$$

Here, $\bar a_-^c$ denotes the mean action of group $c$ from the previous iteration. The newly sampled intended actions are used to update the group mean actions, and this process continues until the joint decisions become stable. The agents then execute their final actions.

{% include figure.liquid path="assets/img/MRG/learning-framework.png" class="img-fluid rounded z-depth-1 d-block mx-auto" width="75%" zoomable=true alt="Mean-field deep reinforcement learning framework" %}

> Building on this framework, we can extend the formulation to other networked systems with heterogeneous agents and develop the corresponding learning algorithms. It can also support further extensions, such as navigation-information design when vehicles do not fully follow the provided guidance and joint adaptive routing under partial observability.

---
{% nocite yang2024markov %}
