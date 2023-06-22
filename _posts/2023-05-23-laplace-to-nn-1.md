---
layout: post
title: "From Laplace to Neural Networks (Part 1)"
# use_math: true
# comments: true
tags: 'NeuralNetworks'
# excerpt_separator: <!--more-->
# sticky: true
# hidden: true
---

Time-series predictions have been always interesting to me (and it's been very hard for me), but I realized that I had never thought about predicting the time-series of physical system dynamics. So I looked up something from my old physics books and gave it a shot.

# Laplace's view

Laplace embraced the view[^1] that if we know the positions and velocities of all the particles in the universe, then we would know the future for all time. 

> *"We may regard the present state of the universe as the effect of its past and the cause of its future. An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed, if this intellect were also vast enough to submit these data to analysis, it would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for such an intellect nothing would be uncertain and the future just like the past would be present before its eyes."*
> 
> -- Pierre Simon de Laplace, A Philosophical Essay on Probabilities[^2]

This *determinism* has been later challenged by a series of development in modern science, such as irreversibility in quantum mechanics or thermodynamics, chaos theory, etc. It can still be a philosophical question[^3], but one thing is clear; even assuming everything is deterministic, nothing is linear in nature. Most of the real problems are highly nonlinear, or I'd say, *chaotically deterministic* or *deterministically chaotic*. 

Nonlinear problems are much harder to solve and they are often high-dimensional (e.g., computational structural biology
, fluid dynamics, material research, etc.), which requires massive computations. Thus, a traditional approach to study physical systems encounters many challenges. Recently, some (frustrated) physicists have found machine learning as a hopeful alternative.

The key questions that physicists have for "machine" are:

>1. Can it *learn* the law of nature only from data?
>2. Can it *predict* chaotic behaviors or nonlinear dynamics?

To me, this sounds a bit crazy - physicists introduced model (i.e., PDEs) to describe the law of nature, and now they're trying to do something in opposite ways? But that is actuallly happening now - [ML for Physics](https://mpl.mpg.de/divisions/marquardt-division/machine-learning-for-physics-science-and-artificial-scientific-discovery) or [Physics for ML](https://www.nature.com/articles/s42254-021-00314-5). This kinds of research evolution is pretty captivating, which motivated me to write this post. As a soft starter, here I consider classical pendulum problems.

# Plane (Ideal) Pendulum

Plane (ideal) pendulum is the most popular example in classical physics and its study is a key to understanding the nonlinear dynamics of many other systems[^4]. 

## Single Pendulum

Consider a particle of mass $m$ constrained by a weightless, extensionless rod to move in a vertical circle of radius $L$ .

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_single.png?raw=true" width="250rem">

 The gravitational force acts downward, but the component of this force influencing the motion is perpendicular to the support rod, and it is simply

$$F(\theta)=-mg\sin\theta$$

The plane pendulum is a nonlinear system with a symmetric restoring force. The equation of motion can be obtained by equating the torque ($=LF(\theta)$) about the support axis to the product of the angular acceleration ($\ddot{\theta}$) and the rotational inertia ($I=mL^2$) about the same axis:

$$ I\ddot{\theta}=LF$$

or,

$$ \ddot{\theta} + w_0^2\sin\theta=0$$

with $w_0^2=\cfrac{g}{L}$.
Thus, if we define new variable $u = [\theta, \dot\theta]^T$, we can construct a system of two first order differential equations that we can solve numerically:

$$
\begin{aligned}
u_1 &= \theta, \ \ \dot u_1 = \dot\theta\\
u_2 &= \dot\theta, \ \ \dot u_2 = \ddot\theta=-\cfrac{g}{L}\sin\theta
\end{aligned}
$$

{% highlight python %}
def equation_motion(self, u, t) -> List[float]:
    """Equations of motion."""
    [theta, theta_dot] = u
    return [theta_dot, -self.g / self.L1 * np.sin(theta)]
{% endhighlight %}

We are going to use [scipy.intergrate.odeint](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html)[^5] to find the numerical solutions.

{% highlight python %}
def solve_ode(self) -> npt.NDArray[Any]:
    """Solve ODE."""
    sol = odeint(
        func=self.equation_motion,
        y0=self.init_cond.u0,
        t=self.time_coord.t_grid,
    )
    return sol
{% endhighlight %}

<br>
Here is the solution (with $m=1.5, L=1, \theta(0)=\frac{80}{180}\pi, \dot\theta(0)=0$). Note that the total energy is preserved over time.

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/single_pendulum.gif?raw=true" width="550rem">

The time-series of $\theta$, $\dot\theta$, $x$, and $y$ are periodic curves:

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/single_pendulum_time_series.png?raw=true" width="800rem">

We often look at its phase diagram and trajectories:

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/single_pendulum_phase_diagram.png?raw=true" width="800rem">



Note that there is an exact analytical solution for this problem, but it is in the form of the Jacobi elliptic integral, and one can skip the details for now.
<details close>
<summary><span style="color: #4a9ae1;">:: Exact solution</span></summary>
<br>
If the amplitude of the motion is small, we may approximate $\sin(\theta)\approx \theta$ , and the above equation becomes a simpler one: $$ \ddot{\theta} + w_0^2\theta=0$$If the amplitude of the motion is not small, we need to use the fact that the system is conservative:$$T +U = E = constant$$The kinetic and potential energies are:$$
\begin{aligned}
T &= \cfrac{1}{2}Iw^2=\cfrac{1}{2}mL^2\dot\theta^2\\
U &= mgL(1-\cos\theta)
\end{aligned}
$$ At the highest point of the motion, we let $\theta = \theta_0$ , then $$
\begin{aligned}
T (\theta_0) &= 0\\
U (\theta_0) &= mgL(1-\cos\theta_0) = E
\end{aligned}
$$. Thus, expressing the kinetic energy as $T=E-U$ , we can see that
$$
\cfrac{1}{2}mL^2\dot\theta^2 = 2mgL\Big[\sin^2(\theta_0/2)-\sin^2(\theta/2)\Big]
$$or, $$
\dot\theta = 2 \sqrt{\cfrac{g}{L}}\Big[\sin^2(\theta_0/2)-\sin^2(\theta/2)\Big]^{1/2}
$$The period can be obtained by the following elliptic integral (of the first kind):
$$
\tau = 2 \sqrt{\cfrac{L}{g}}\displaystyle\int_0^{\theta_0} \Big[\sin^2(\theta_0/2)-\sin^2(\theta/2)\Big]^{-1/2}d\theta
$$
</details>


## Double Pendulum

Double pendulum consists of two masses and two rods. The situation is slightly more complicated, since the two masses interact with each other. 

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/diagram_double.png?raw=true" width="250rem">

But using the Euler-Lagrangian equation[^6], we can obtain a system of equations of motion and solve it numerically.

<details close>
<summary><span style="color: #4a9ae1;">:: How?</span></summary>
<br>
The cartesian locations of the two masses are:

$$\begin{aligned}
x_1 &= L_1\sin(\theta_1)\\
x_2 &= L_1\sin(\theta_1) + L_2\sin(\theta_2)\\
y_1 &= -L_1\cos(\theta_1)\\
y_2 &= -L_1\cos(\theta_1) - L_2\cos(\theta_2)
\end{aligned}
$$

The potential energy:

$$\begin{aligned}
U &= m_1gy_1 + m_2gy_2\\
&=-m_1gL_1\cos(\theta_1)-m_2g\big[L_1\cos(\theta_1) + L_2\cos(\theta_2)\big]
\end{aligned}
$$

The kinetic energy:

$$
\begin{aligned}
T &= \cfrac{1}{2}m_1(\dot{x}_1^2 + \dot{y}_1^2) + \cfrac{1}{2}m_2(\dot{x}_2^2 + \dot{y}_2^2) \\
&= \cfrac{1}{2}m_1\dot{\theta}_1^2L_1^2 + \cfrac{1}{2}m_2\big[\dot{\theta}_1^2L_1^2 + \dot{\theta}_1^2L_1^2 + 2\theta_1\theta_2L_1L_2\cos(\theta_1-\theta_2)\big]
\end{aligned}
$$

Note that the Lagrangian of a system,

$$ L=T-U$$

must follow Euler-Lagrangian differential equation:


$$ \cfrac{d}{dt}\Bigg(\cfrac{\partial L}{\partial \dot{\theta}}\Bigg)-\cfrac{\partial L}{\partial \theta} = 0
$$

In our case, the Lagrangian is

$$
\begin{aligned}
L &= T - U\\
&= \cfrac{1}{2}m_1\dot{\theta}_1^2L_1^2 + \cfrac{1}{2}m_2\big[\dot{\theta}_1^2L_1^2 + \dot{\theta}_1^2L_1^2 + 2\theta_1\theta_2L_1L_2\cos(\theta_1-\theta_2)\big]\\
&+m_1gL_1\cos(\theta_1)+m_2g\big[L_1\cos(\theta_1) + L_2\cos(\theta_2)\big]
\end{aligned}
$$

Thus, the Euler-Lagrangian differential equations for ($\theta_1$, $\dot\theta_1$) and ($\theta_2$, $\dot\theta_2$) will give us the solutions for $\ddot\theta_1$ and $\ddot\theta_2$, respectively: 

$$
\begin{aligned}
\ddot{\theta}_1&=\cfrac{m_2g\sin(\theta_2)\cos(\theta_1-\theta_2)-m_2\sin(\theta_1-\theta_2)\big[L_1\cos(\theta_1-\theta_2)\dot\theta_1^2+L_2\dot\theta_2^2\big]}{L_1\big(m_1+m_2\sin(\theta_1-\theta_2)\big)}\\

\ddot{\theta}_2&=\cfrac{(m_1+m_2)\Big[L_1\dot\theta_1^2\sin(\theta_1-\theta_2) - g\sin(\theta_2) + g\sin(\theta_1)\cos(\theta_1-\theta_2)\Big] + m_2L_2\dot\theta_2^2\sin(\theta_1-\theta_2)\cos(\theta_1-\theta_2)}{L_1\big(m_1+m_2\sin^2(\theta_1-\theta_2)\big)}
\end{aligned}
$$


$$
\begin{aligned}
u_1 &= \theta_1, \dot u_1 = \dot\theta_1\\
u_2 &= \dot\theta_1, \dot u_2 = \ddot\theta_1\\
u_3 &= \theta_2, \dot u_3 = \dot\theta_2\\
u_4 &= \dot\theta_2 , \dot u_4 = \ddot\theta_2\\
\end{aligned}
$$

where $\ddot\theta_1$ and $\ddot\theta_2$ are given above. This can be also solved numerically.

</details>
<br>

The equation of motions are implemented and solved similarly, and the solved results are in the below:

(with $m_1=1.5, m_2=2, L_1=1, L_2=1.5, \theta_1(0)=\frac{60}{180}\pi, \theta_2(0) = \frac{80}{180}\pi, \dot\theta_1(0)=\dot\theta_2(0)=0$)

<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/double_pendulum.gif?raw=true" width="550rem">

The time series of ($\theta_1$, $\dot\theta_1$), ($\theta_2$, $\dot\theta_2$), ($x_1$, $y_1$), and ($x_2$, $y_2$) are not periodic anymore:
<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/double_pendulum_time_series.png?raw=true" width="800rem">

The phase diagram and trajectories shows the coupling relation between $\theta_1$ and $\theta_2$. 
<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/double_pendulum_phase_diagram.png?raw=true" width="800rem">


# Predicting pendulum *time-series* - ARIMA?

So far we have seen that the single pendulum exhibits periodic dynamics, while the double pendulum exhibits more chaotic and nonlinear dynamics. 

Here we attempt to predict the time-series of pendulum phase states. [**ARIMA**](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) is one of the most commonly used methods to model univariate time-series. Its extension to reflect seasonality is SARIMA. 

For simplicity, we focus on predicting $\theta$ of single pendulum and $\theta_1$ of double pendulum. The time-series data is splitted into training and test region, and the predicted values are compared with the actuals in the both regions:

- In-sample: both ARIMA and SARIMA work fine.
- Out-of-sample: ARIMA fails for single/double pendulums. SARIMA works for single pendulum, but not for double pendulum. I further conducted a [grid search](https://github.com/uriyeobi/PIML/blob/main/notebooks/pendulum_arima.ipynb) for parameters, but couldn't find a SARIMA model that works well for double pendulum predictions.


<img src="https://github.com/uriyeobi/PIML/blob/main/notebooks/fig/pendulum_arima.png?raw=true" width="800rem">


In short, for our double pendulum, (S)ARIMA seems to be fine for curve-fitting, but not for predictions. It is not surprising, since (S)ARIMA cannot approximate all complex dynamics and generalize. (S)ARIMA was tried just as a warm-up.

We want to try something else.

# What's Next

In Part 2, we'll adopt neural networks to predict pendulum dynamics. Particularly, we will see if it learns any physics purely from data and how embedding physics helps it learn better.


<br>

****


**Codes**: [https://github.com/uriyeobi/PIML](https://github.com/uriyeobi/PIML)

**Notes**

[^1]: [Laplace's Demon](https://en.wikipedia.org/wiki/Laplace%27s_demon)

[^2]: [A Philosophical Essay on Probabilities (Translated to English)](https://bayes.wustl.edu/Manual/laplace_A_philosophical_essay_on_probabilities.pdf)

[^3]: free will, memory, observation, etc.

[^4]: Classical Dynamics of Particles and Systems (5th Edition), Thornton and Marion, Brooks/Cole 2004

[^5]: There are other packages such as [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), which is for more general options for integrators.

[^6]: [https://mse.redwoods.edu/darnold/math55/DEproj/sp08/jaltic/presentation.pdf](https://mse.redwoods.edu/darnold/math55/DEproj/sp08/jaltic/presentation.pdf)
