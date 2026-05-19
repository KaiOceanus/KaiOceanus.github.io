## KL散度

在概率论和信息论中，**KL 散度**（Kullback-Leibler Divergence），又称**相对熵**（Relative Entropy），用于衡量两个概率分布 _P_ 和 _Q_ 之间的差异程度，衡量用分布Q近似分布P时的信息损失。$KL(P||Q) >=0$ ，当且仅当P = Q时等号成立。

- $P$ 通常代表数据的**真实分布**（Target distribution）。
- $Q$ 通常代表数据的**理论分布**或模型的**近似分布**（Approximating distribution）。

离散：

$$
D_{KL}(P \parallel Q) = \sum_{i} P(i) \log \frac{P(i)}{Q(i)}
$$

连续：

$$
D_{KL}(P \parallel Q) = \int p(x) \log \frac{p(x)}{q(x)} dx
$$

## PPO

目标：往人类打分高的方向输出

1. 收集人类反馈
2. **训练奖励模型**
3. 使用PPO训练策略

## 思考题

### DPO

#### DPO训练出现过拟合的原因

### PPO

#### PPO训练中奖励坍塌的原因及如何改善

## 参考资料

[从零实现PPO算法](https://www.bilibili.com/video/BV1CHQDYHEsU)

[强化学习的数学原理](https://www.bilibili.com/video/BV1sd4y167NS)
