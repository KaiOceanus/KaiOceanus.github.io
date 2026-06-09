# PPO算法原理及实现

# PPO (Proximal Policy Optimization)

## 1. PPO Clipped Objective

$$
L^{CLIP}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right]
$$

其中

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{old}}(a_t|s_t)}
$$

---

## 2. PPO 总损失函数

$$
L(\theta)
=
L^{CLIP}(\theta)
-
c_1 L^{VF}(\theta)
+
c_2 S[\pi_\theta]
$$

---

## 3. Value Function Loss

$$
L^{VF}(\theta)
=
\mathbb{E}_t
\left[
(V_\theta(s_t)-V_t^{target})^2
\right]
$$

---

## 4. Entropy Bonus

$$
S[\pi_\theta]
=
\mathbb{E}_t
\left[
-\sum_a
\pi_\theta(a|s_t)
\log \pi_\theta(a|s_t)
\right]
$$

---

## 5. Return

$$
R_t
=
\sum_{k=0}^{\infty}
\gamma^k r_{t+k}
$$

---

## 6. Advantage Function

最简单形式：

$$
\hat A_t
=
R_t - V(s_t)
$$

---

## 7. GAE (Generalized Advantage Estimation)

TD Error：

$$
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
$$

GAE：

$$
\hat A_t^{GAE}
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}
$$
