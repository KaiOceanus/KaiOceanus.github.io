# GRPO算法原理及实现

GRPO算法（Group Relative Policy Optimization）2024年由deepseek在论文[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)中提出，对PPO算法做了如下改进：

1. 去掉Critic Model，不再用GAE计算优势
2. 使用对同一问题采样得到的多个输出的**平均奖励**作为baseline
3. KL散度直接加入到loss中，而不是加入奖励中作为惩罚项

## GRPO的目标函数

$$
J_{GRPO}(\theta)
=
\mathbb{E}
\left[
\frac1G
\sum_{i=1}^{G}
\min
\Big(
\rho_i(\theta)A_i,
\;
\operatorname{clip}
(
\rho_i(\theta),
1-\epsilon,
1+\epsilon
)
A_i
\Big)
\right]
$$

引入$\pi_{ref}$和KL散度：
$$
L(\theta)
=
J_{GRPO}(\theta)
-
\beta
D_{KL}
(
\pi_\theta
\|
\pi_{ref}
)
$$
