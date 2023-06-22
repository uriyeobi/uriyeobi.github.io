---
layout: post
title: "From Laplace to Neural Networks (Part 2)"
# use_math: true
# comments: true
tags: 'NeuralNetworks'
# excerpt_separator: <!--more-->
# sticky: true
# hidden: true
---


We continue the discussion in [Part 1](https://uriyeobi.github.io/2023-05-23/laplace-to-nn-1), but now using neural networks. Can neural networks really predict a nonlinear system like double pendulum?

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/double_pendulum.gif?raw=true" width="400rem">


Well, we know from [universal approximation theorem of neural networks](https://en.wikipedia.org/wiki/Universal_approximation_theorem) that neural networks can approximate any smooth function. So given time-series data of the pendulum's positions and velocities, which is governed by differential equations, we may train neural networks and forecast future dynamics.

This post is particularly motivated by a recent [paper](https://arxiv.org/abs/1906.01563), which exploits the concepts of Hamiltonian mechanics.

---
<br>

# Hamiltonian Mechanics
Hamiltonian of a system is expressed by [generalized coordinates $q$ and momenta $p$](https://en.wikipedia.org/wiki/Generalized_coordinates):

$$
\begin{aligned}
\mathcal{H} &= \mathcal{H}(q_i, p_i, t)
\end{aligned}
$$

The crux of Hamiltonian mechanics is **Hamilton's equations of motions**, which are given by[^1]:

$$
\begin{aligned}
\dot q_i &= \cfrac{\partial\mathcal{H}}{\partial p_i}\\
\dot p_i &= -\cfrac{\partial\mathcal{H}}{\partial q_i}
\end{aligned}
$$


These relations have a special implication about the total energy of the system. Consider the total differential of $H$:

$$
\begin{aligned}
dH = \sum_i\Big(\cfrac{\partial\mathcal{H}}{\partial q_i}dq_i + \cfrac{\partial\mathcal{H}}{\partial p_i} dp_i\Big)
 + \cfrac{\partial H}{\partial t} dt
\end{aligned}
$$

With Hamilton's equations, the term in the parenthesis vanishes, and it follows that

$$
\begin{aligned}
\cfrac{d\mathcal{H}}{dt} = \cfrac{\partial \mathcal{H}}{\partial t}
\end{aligned}
$$

Thus, if $$\mathcal{H}$$ does not have an explicit dependency on $t$, then the Hamiltonian is a conserved quantity. 

Let's take a few examples.

## Single Pendulum 


<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_single.png?raw=true" width="250rem">

In the case of single pendulum, the Hamiltonian is given as

$$
\begin{aligned}
\mathcal{H} &=T+U\\
&=\cfrac{1}{2}mL^2\dot\theta^2 - mgL\cos\theta\\
&=\cfrac{p^2}{2mL^2} - mgL\cos(q)
\end{aligned}
$$

where we defined the generalized coordinates $$q:=\theta$$ and generalized momenta $$p:=mL^2\dot\theta$$ [^2].

Hamilton's equations of motions are:

$$
\begin{aligned}
\dot q &= \cfrac{\partial\mathcal{H}}{\partial p} = \cfrac{p}{mL^2}\\
\dot p &= -\cfrac{\partial\mathcal{H}}{\partial q} = -mgL\sin q
\end{aligned}
$$

## Double Pendulum 


<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_double.png?raw=true" width="250rem">


Similarly, the Hamiltonian can be expressed as

$$
\begin{aligned}
H(q_1,q_2,p_1,p_2) &= \cfrac{m_2L_2^2p_1^2 + (m_1+m_2)L_1^2p_2^2 - 2m_1L_1L_2p_1p_2\cos(q_1-q_2)}{2m_2L_1^2L_2^2\big[m_1+m_2\sin^2(q_1-q_2)\big]}\\
&-(m_1+m_2)gL_1\cos(q_1) - m_2gL_2\cos(q_2)
\end{aligned}
$$

Hamilton's equations of motions are:

$$
\begin{aligned}
\dot q_1 &= \cfrac{\partial H}{\partial p_1} = \cfrac{L_2p_1 - L_1p_2\cos(q_1-q_2)}{L_1^2L_2\big[m_1+m_2\sin^2(q_1-q_2)\big]}\\
\dot q_2 &= \cfrac{\partial H}{\partial p_2} = \cfrac{-m_2L_2p_1\cos(q_1-q_2) + (m_1+m_2)L_1p_2}{m_2L_1L_2^2\big[m_1+m_2\sin^2(q_1-q_2)\big]}\\
\dot p_1 &= - \cfrac{\partial H}{\partial q_1} = -(m_1+m_2)gL_1\sin(q_1) - h_1 + h_2 \sin[2(q_1-q_2)]\\
\dot p_2 &= - \cfrac{\partial H}{\partial q_1} = -m_2gL_2\sin(q_2) + h_1 - h_2\sin[2(q_1-q_2)]
\end{aligned}
$$

where:

$$
\begin{aligned}
h_1 &:= \cfrac{p_1p_2\sin(q_1-q_2)}{L_1L_2\big[m_1+m_2\sin^2(q_1-q_2)\big]}\\
h_2 &:= \cfrac{m_2L_2^2p_1^2 + (m_1+m_2)L_1^2p_2^2 - 2m_2L_1L_2p_1p_2\cos(q_1-q_2)}{2L_1^2L_2^2\big[m_1+m_2\sin^2(q_1-q_2)\big]^2}
\end{aligned}
$$

# Hamiltonian Neural Networks (HNN)

The key idea of HNN is *learning a Hamiltonian from data*. The neural networks model consumes generalized coordinates ($q$) and momenta ($p$) and produces a single "energy-like" scalar. Its training is based on the loss function that is a L2 discrepancy of Hamilton's equations:

$$
\begin{aligned}
L_{HNN}\big(H(q,p), \dot q, \dot p\big) = \Bigg\lVert\dot q - \cfrac{\partial\mathcal{H}}{\partial p} \Bigg\lVert_2^2 + \Bigg\lVert\dot p + \cfrac{\partial\mathcal{H}}{\partial q}\Bigg\lVert_2^2
\end{aligned}
$$

This approach is different from the baseline benchmark such as multilayer perceptrons, which learns the time-derivatives ($\dot q$ and $\dot p$) directly. 

A striking thing about HNN is that even without hand-crafting the Hamiltonian in the model, the neural networks learn the Hamiltonian directly from data. Note that the architecture depicted above does not include anything about Hamiltonian - all we designed in HNN is to output a certain scalar ($\mathcal{H}$), and use the above loss function based on Hamilton's equations. Nevertheless, the total-energy is preserved well in the system that is predicted by HNN models.

<br>

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_base.png?raw=true" width="650rem">
<center>[Baseline Multilayer Perceptrons] </center><br>

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_hnn.png?raw=true" width="800rem">
<center>[Hamiltonian Neural Networks] </center>

<br>


What if we feed physics knowledge more explicitly to the training proces? Yes, it is likely to learn even better. An additional penalty can be imposed as below.

$$
\begin{aligned}
L_{HNN} &= L_{data} + L_{physics} = \Bigg\lVert\dot q - \cfrac{\partial\mathcal{H}}{\partial p} \Bigg\lVert_2^2 + \Bigg\lVert\dot p + \cfrac{\partial\mathcal{H}}{\partial q}\Bigg\lVert_2^2 + \lambda\Big\lVert H( q,  p) - H_0 \Bigg\lVert_2^2
\end{aligned}
$$

where $H_0$ is a constant, the total-energy of the system measured from the initial condition.


# Implementation of HNN

The codes are available in [github.com/uriyeobi/PIML](https://github.com/uriyeobi/PIML). The implementation is inspired by that of the original paper where the authors use `pytorch`. Here I simplified and redesigned it in `tensorflow.keras`. I also added the energy-conservation parts and the task for double pendulum, which are not included in the paper. 

As seen from the code below, the output dimension of HNN is 1 (for the scalar $H$), and the loss from Hamilton's equations ($L_{data}$) is calculated by using the gradients (from `forward` method) and permutation matrix (`M`). For activation, `tanh` is used:

{% highlight python %}class HNN(tf.keras.Model):
    """Hamiltonian Neural Networks."""

    def __init__(self, input_dim, hidden_dims, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = tf.keras.Sequential(
            [tf.keras.Input(shape=(input_dim,))]
            + [
                tf.keras.layers.Dense(hidden_dim, activation="tanh")
                for hidden_dim in hidden_dims
            ]
        )

        self.last_layer = tf.keras.layers.Dense(1)
        self.input_dim = input_dim

    @cached_property
    def M(self) -> npt.NDArray[Any]:
        """Permutation matrix (assuming canonical coordinates)."""
        M = np.eye(self.input_dim)
        M = np.concatenate(
            (M[self.input_dim // 2 :], -M[: self.input_dim // 2]), axis=0
        )
        return tf.constant(M, dtype="double")

    def call(self, x):
        """Call."""
        features = self.feature_extractor(x)
        outputs = self.last_layer(features)
        return outputs

    def forward(self, x):
        """Forward."""
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x)
            outputs = self.last_layer(features)
        return (tape.gradient(outputs, x)) @ self.M


{% endhighlight %}

As for the loss function implemtation, we pre-calculate the initial Hamiltonian (`ham0`) and compare it with the new estimated Hamiltonian, throughout the training process. Then the loss for the energy conservation, $L_{physics}$, is added to the loss for Hamilton's equations, $L_{data}$:

{% highlight python %}def get_loss(model, x, y, ham0, mlg, hamiltonian_method, penalty_lamb):
    """Get loss."""
    predictions = model.forward(tf.Variable(tf.stack(x)))
    ham_new = get_hamiltonian(
        x=x, mlg=mlg, hamiltonian_method=hamiltonian_method, predictions=predictions
    )

    physics_embedded_penalty = tf.reduce_mean(tf.square(ham0 - ham_new))

    return (
        tf.reduce_mean(tf.square(predictions - tf.Variable(tf.stack(y))))
        + penalty_lamb * physics_embedded_penalty
    )
{% endhighlight %}
<br>



# Performance of HNN

The goal is to predict the time-series of generalized coordinates and momenta for the out-of-sample test data[^6].

Here is sample data:
- Single pendulum: $q, p, \dot q, \dot p$
- Double pendulum: $q_1, q_2, p_1, p_2, \dot q_1, \dot q_2, \dot p_1, \dot p_2$

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/pendulum_data_sample2.png?raw=true" width="800rem">

The hyperparameters[^3] are set differently for single and double pendulums.

|          Hyperparameters         | Singe Pendulum | Double Pendulum |
|:--------------------------------:|:--------------:|:---------------:|
|      Number of Hidden Layers     |        2       |        4        |
| Number of Nodes in Hidden Layers |       200      |       400       |
|           Learning Rate          |      0.002     |      0.00025    |
|              $\lambda$           |      0.1       |       0.1       |
{:.type1}

The out-of-sample performances are measured for the following three models:

- `base`: Base MLP
- `hnn0`: HNN with $\lambda=0$ ($L_{data}$)
- `hnn1`: HNN with $\lambda=0.1$ ($L_{data}+L_{physics}$)


### Single Pendulum

Here are the results in the test dataset: time-series of $q, p$ and their errors. HNNs are much better than the base MLP. 

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_single_pendulum_pred_act.png?raw=true" width="800rem">


<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_single_pendulum_error_qp.png?raw=true" width="800rem">

Here are the errors in total energy (Note that y-axis is in log-scale). The energy-conservation is slightly better in `hnn1`, compared to `hnn1`.

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_single_pendulum_error_energy.png?raw=true" width="600rem">

<br>

For the single pendulum, both `hnn0` and `hnn1` predict the time-series pretty well. 


### Double Pendulum

Here is the predicted and actual time-series of $q1,q2,p1,p2$. As expected, it is not as perfect as the single pendulum. 

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_double_pendulum_pred_act.png?raw=true" width="800rem">

We observe that `hnn1` definitely outperforms the others (`hnn0` is not necessarily better than `base`). Recall that these plots are out-of-sample forecasts starting from time step = 1600, so the predictions become less accurate as the time step increases[^5].

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_double_pendulum_error_qp.png?raw=true" width="800rem">


Lastly, the impact of embedding physics on the performance (i.e., adding the energy conservation penalty $L_{physics}$) is demonstrated in the following energy error plot - it helps conserve the total energy of the forecasted system!

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/hnn_double_pendulum_error_energy.png?raw=true" width="600rem">


# Summary

In this and [previous](https://uriyeobi.github.io/2023-05-23/laplace-to-nn-1) posts, we took the case of pendulum systems and explored the possibility of neural networks approaches to predict the non-linear dynamics in the out-of-sample region. In particular, we focused on the idea of Hamiltonian Neural Networks (HNN). Hamilton's equations of motions represent very generic relations in physical systems, so the neural networks discover the physics from data. Consequently, the HNN works quite well in predicting the complex dynamics of the double pendulum. We also confirmed that the embedded physics[^4] in the model boosts up the performance.

Although HNN is a powerful tool, there is a wrinkle in it - the Hamiltonian formalism requires that the coordinates of the system be ["canonical"](https://en.wikipedia.org/wiki/Canonical_transformation), but many systems do not necessarily satisfy that condition. This motivated researchers to devise alternatives such as [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630), which enables the model to learn with arbitrary coordinates.

<br>

****


**Codes**: [https://github.com/uriyeobi/PIML](https://github.com/uriyeobi/PIML)

**Notes**


[^1]: [Hamiltonian Mechanics - Wikipedia](https://en.wikipedia.org/wiki/Hamiltonian_mechanics).

[^2]: The generalized momenta is given by a partial derivative of the Lagrangian ($L = T - U$) with respect to $$\dot q$$.

[^3]: The hyperparameters were chosen from a grid search - number of nodes: [100, 200, 400], number of hidden layers: [2, 3, 4], learning rate: [0.0005, 0.001, 0.002], $\lambda$: [0.1, 0.05]. There could be improved performance from further fine tuning of hyperparameters, but that's not the main scope of this work.

[^4]: [Physics-Informed Machine Learning (PIML)](https://www.nature.com/articles/s42254-021-00314-5) (or [Physics-Informed Neural Networks (PINN)](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)) is relatively new, but emerging very rapidly recently, in almost all applications (to name a few - [fluid dynamics](https://link.springer.com/article/10.1007/s10409-021-01148-1), [power systems](https://ieeexplore.ieee.org/abstract/document/9282004), [climate](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0093), [manufacturing](https://www.sciencedirect.com/science/article/pii/S0278612521002259?casa_token=DBELy1Iq6eUAAAAA:jo8CnbdP1yw04kpQa4QSXDlcF--Zne78ZG3Bwz8Q-TU8t_dcGiPi15bu1JwcDpLuJaCHUhvL4A), [heat transfer](https://asmedigitalcollection.asme.org/heattransfer/article/143/6/060801/1104439/Physics-Informed-Neural-Networks-for-Heat-Transfer), etc.)

[^5]: There could be several components for the errors. For their decomposition, see [https://arxiv.org/pdf/2202.04836.pdf](https://arxiv.org/pdf/2202.04836.pdf).

[^6]: 4:1 ratio for train and test data split.