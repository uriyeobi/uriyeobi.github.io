---
layout: post
title: "Statistical Mechanics and Statistical Inference"
# use_math: true
# comments: true
tags: 'Physics StatisticalInference'
# excerpt_separator: <!--more-->
# sticky: true
# hidden: true
---

I have to confess that when I was a physics student, I thought taking classes on probability theory and statistics was an unnecessary distraction from learning "real physics". But while reading ML/stats literature, I have witnessed its extensive connection with physics, especially statistical mechanics. This note scratches some surface of it.

---


<br>


Let's start out with a simple setup.

# Counting particles freely - uniform distribution

Consider a fixed volume $V$ with $N$ particles that don't interact with each other. Each particle is described by its position and velocity vectors.

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/posts_stats_mech_stats_inf/particles.png?raw=true" width="350rem">

Now we do an experiment of distributing $N$ particles into $M$ cells in total. We denote the number of particles as $n_i$, for $i=1, 2, \dots, M$. 


$$
\begin{aligned}
\displaystyle\sum_{i} n_i &= N\\
\end{aligned}
$$

What we are interested in here is:

> What is the **most likely distribution** of $N$ particles?

When $N$ is large, there are (so) many possibilities of how particles are distributed. The number of all possible combinations is:

$$\Omega (\mathbf{n})= \cfrac{N!}{n_1!n_2!\cdots n_M!}$$

where $\mathbf{n} = [n_1, n_2, \dots, n_M]$ is a vector of configuration, $n_i$. It is not hard to see that the most probable macrostate is the one with maximum $\Omega$. 

<details close>
<summary><span style="color: #4a9ae1;">:: why maximum $\Omega$?</span></summary>
<br>

The average probability of being found in state $j$ is give by

$$p_j = \cfrac{\overline{n_j}}{N} = \frac{1}{N}\cfrac{\displaystyle\sum_{\mathbf{n}}n_j(\mathbf{n})\Omega(\mathbf{n})}{\displaystyle\sum_{\mathbf{n}}\Omega(\mathbf{n})}$$

But the function $\Omega(\mathbf{n})$ has a very sharp peak at its maximum, when $n_i$'s are large (this is the regime where we live). Thus, the above probability can be approximated by 

$$p_j \approx \cfrac{1}{N}\cfrac{n_j(\mathbf{n}^*)\Omega(\mathbf{n}^*)}{\Omega(\mathbf{n}^*)} = \cfrac{n_j(\mathbf{n}^*)}{N}$$

where $\mathbf{n}^* = \underset{\mathbf{n}}{\mathrm{argmin}}\Omega(\mathbf{n})$. So we can just focus on finding $\mathbf{n}^*$.

</details>

<br>

Without further information, $\Omega(\mathbf{n})$ is maximized when the distribution of particles is uniform: 

$$p_i = \cfrac{n_i}{N} = \cfrac{1}{M}$$

Although it is not hard to show mathematically why the uniform distribution is the answer here, I would take this as very intuitive - [above all, why not uniform?](https://en.wikipedia.org/wiki/Principle_of_indifference).

# Counting particles with fixed temperature - Boltzmann distribution

Now, we have additional information (constraint) that the temperature is fixed[^4], so the total energy ($E$) of the system is constant. Denoting the energy state for each cell as $\epsilon_i$ , respectively, for $i=1, 2, \dots, M$, we now have the two constraints: 

$$
\begin{aligned}
\displaystyle\sum_{i} n_i &= N\\
\displaystyle\sum_{i} n_i \epsilon_i &= E \ \ (\because \textsf{non interacting particles})
\end{aligned}
$$

Similar to the previous case, the most likely distribution maximizes $\Omega$. To find it, we need to solve the following optimization (Sometimes it is easier to work with "$\log$"):

$$
\begin{aligned}
&\max_{\mathbf{n}}\log\Omega(\mathbf{n})\\
\textrm{s.t.}& \sum_i n_i = N\\
& \sum_i \epsilon_i n_i = E
\end{aligned}
$$

First, observe that, using the [Stirling's formula](https://en.wikipedia.org/wiki/Stirling%27s_approximation):

$$
\begin{aligned}
\log\Omega &= \log N! - \displaystyle\sum_{i} \log n_i!\\
&= N\log N - N - \displaystyle\sum_{i} \big[ n_i \log n_i - n_i \big]\\
&= N\log N - \displaystyle\sum_{i}  n_i \log n_i
\end{aligned}
$$

Second, at the maximum of $\log \Omega$, the derivative with respect to $n_i$ must vanish, so we take differential:

$$
\begin{aligned}
0 &= \delta (-\log \Omega(\mathbf{n})) \\
&= \delta \displaystyle\sum_{i} n_i \log n_i \\
&= \displaystyle\sum_{i} \big[n_i\delta \log n_i + \delta n_i \log n_i \big] \\
&= \displaystyle\sum_{i} \delta n_i +\displaystyle\sum_{i} \delta n_i \log n_i
\end{aligned}
$$

In addition, since the $N$ and $E$ are constants, we have:

$$
\begin{aligned}
\delta N &= \displaystyle\sum_{i} \delta n_i = 0\\
\delta E &= \displaystyle\sum_{i} \epsilon_i \delta n_i = 0
\end{aligned}
$$

Using the method of [Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier), 

$$
\begin{aligned}
0 &= \delta \log \Omega(\mathbf{n}) + \alpha \displaystyle\sum_{i} \delta n_i + \beta \displaystyle\sum_{i} \epsilon_i \delta n_i\\
&= \displaystyle\sum_{i} \delta n_i +\displaystyle\sum_{i} \delta n_i \log n_i + \alpha \displaystyle\sum_{i} \delta n_i + \beta \displaystyle\sum_{i} \epsilon_i \delta n_i \\
&= \displaystyle\sum_{i}  \delta n_i \Big[1 + \log n_i + \alpha + \beta \epsilon_i \Big]
\end{aligned}
$$

This yields the so-called Boltzmann factor:

$$n_i = C e^{-\beta \epsilon_i}$$

where $C = e^{-(1+\alpha)}$ is a normalization constant: But since $\displaystyle\sum_{i} n_i=N$, we obtain:

$$C = \cfrac{N}{\displaystyle\sum_{i} e^{-\beta \epsilon_i }}$$

Finally, the probability distribution for the level population is given by [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution):

$$ p_i = \cfrac{n_i}{N} = \cfrac{e^{-\beta \epsilon_i }}{\displaystyle\sum_{i} e^{-\beta \epsilon_i }} $$

This formulation is actually familiar to ML people as [softmax](https://en.wikipedia.org/wiki/Softmax_function). Note that there were no physics postulates used in this derivation - all we did so far were enumeration and solving an optimization problem.

# Entropy and maximum entropy principle

Enumerating states is directly linked to entropy[^1]. 

In statistical mechanics, the entropy is defined as the log of number of states accessible to the system with coefficient $k_B$ ([Boltzmann constant](https://en.wikipedia.org/wiki/Boltzmann_constant))[^7]:

$$
\begin{aligned}
S &= k_B\log (\textsf{number of configurations})\\
&= k_B\log \Omega
\end{aligned}
$$

The more states that are accessible, the greater the entropy.

Now, we see from the above:

$$
\begin{aligned}
\log\Omega &= N\log N - \displaystyle\sum_{i}  n_i \log n_i\\
&= \displaystyle\sum_{i}  n_i \log N - \displaystyle\sum_{i}  n_i \log n_i \ \ (\because \displaystyle\sum_{i}  n_i = N)\\\
&= - \displaystyle\sum_{i}  n_i \log \cfrac{n_i}{N }
\end{aligned}
$$

So if we divide this by $N$, we obtain

$$
\begin{aligned}
\cfrac{1}{N} \log\Omega &= - \displaystyle\sum_{i} \cfrac{n_i}{N}\log \cfrac{n_i}{N}\\
&= - \displaystyle\sum_{i} p_i \log p_i\\
&  \equiv  H(\mathbf{p})
\end{aligned}
$$

where $H(\mathbf{p}) = - \displaystyle\sum_{i} p_i \log p_i $ denotes the [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) in information theory. This indicates that, up to a constant factor, 

> Thermodynamic entropy is identical to information theory entropy. 

What does this imply? It implies opening a broader view in understanding many concepts in statistics or physics. It was actually [E.T. Jaynes](https://en.wikipedia.org/wiki/Edwin_Thompson_Jaynes) who claimed that statistical mechanics does not need to be regarded as a physical theory (which depends on some assumptions such as the ergodic hypothesis) - it can be considered as a form of statistical inference. 

### Maximum Entropy Principle
Note that the Boltzmann distribution is often derived from [Maximum Entropy Principle](https://en.wikipedia.org/wiki/Principle_of_maximum_entropy). That is,

$$
\begin{aligned}
&\max_{\mathbf{p}} H(\mathbf{p})\\
\textrm{s.t.}& \sum_i p_i = 1\\
& \sum_i \epsilon_i p_i = E/N = \langle \epsilon \rangle_{\mathbf{p}}
\end{aligned}
$$

This implies that physical representation in our experiment is just the best estimates we could infer based on the information available. Intuitively, the uniform distribution is perfectly random and thus perfectly unpredictable; more predictable distributions should carry less information. With this in mind, the Principle of Maximum Entropy can further be interpreted as choosing a distribution that is as uniform as possible subject to the given constraints.



For example, how can we interpret Gaussian distribution?

# Gaussian distribution from maximum entropy

Gaussian distribution is used everywhere. But why? Other than its nice and useful mathematical properties, I had not been quite convinced (e.g., what is its *physical meaning*?), but here is its connection with entropy:

> Gaussian distribution is the distribution with maximum entropy given the constraints of a fixed mean and variance.

Suppose $X$ is a continuous random variable with probability density $p(x)$. Then we can define [differential entropy](https://en.wikipedia.org/wiki/Differential_entropy) of $X$ as:

$$H(X) = - \displaystyle\int p(x) \log p(x) dx$$

Suppose we have the following constraints:

$$
\begin{aligned}
\textsf{mean}(X) &= \mu\\
\textsf{var}(X) &= \sigma^2
\end{aligned}
$$


If we consider the following functional form:

$$A =  \int -p(x)\log p(x)dx + \lambda_1\Big[ \int p(x) dx - 1\Big] + \lambda_2\Big[\int (x-\mu)^2 p(x) dx - \sigma^2\Big]$$ 

then, by varying $p(x)$, we find

$$ \log p(x) + 1 - \lambda_1 - \lambda_2 (x - \mu)^2 = 0$$

which yields the Gaussian form:

$$ p(x) = e^{-\lambda_1 + 1 - \lambda_1 (x-\mu)^2}$$

and it is not hard to find $\lambda_1$ and $\lambda_2$, by using the constraints.

This interpretation is also consistent from the perspective of [Central limit theorem](https://en.wikipedia.org/wiki/Central_limit_theorem), which says that piling up *any* random variables converge to Gaussian random variables. Suppose you have data of Gaussian distribution. Then, it is likely that the information about the original distribution has been gone or washed out already, so you don't know which distribution it originally came from. This means that there is less information in Gaussian distribution, which implies higher entropy than other distributions.


# Maximum Entropy = Minimum Free Energy

Entropy is linked to free energy.

### Free energy

The "free" in free energy means "available". Simply put, free energy is the energy available to do work. But what does it actually mean? Physicists tend to describe systems in terms of energy rather than probability, so they come up with something called free energy. A bit more formally, the free energy of a state is that energy whose Boltzmann factor gives the correct relative probability of the state. If the free energy for state $A$ and $B$ are defined as $F_A$ and $F_B$, then it means that

$$ \cfrac{\exp{\Big[-\cfrac{F_A}{k_BT}}\Big]}{\exp{\Big[-\cfrac{F_B}{k_BT}}\Big]}= \cfrac{p_A}{p_B} $$

So the fundamental implication of free energy is most or less *probability*. 

With fixed particle numbers, volume, and temperature, the free energy is referred to as the [Helmholtz free energy](https://en.wikipedia.org/wiki/Helmholtz_free_energy)[^2], which can be stated as:

$$F_{\textsf{Helmholtz}} = E - TS$$

### Entropy Maximization = Free Energy Minimization

The [entropy maximization problem we discussed above](#maximum-entropy-principle) can be converted to a unconstrained minimization problem with the following objective function

$$F(\mathbf{p}) = \langle \epsilon \rangle_{\mathbf{p}} - \cfrac{1}{\beta}H(\mathbf{p})$$

Note that if $\beta = \cfrac{1}{k_BT}$, then the two expressions above are identical:

The first term of the free energy is the internal energy and the second term is the entropy of the distribution $\mathbf{p}$, scaled by $\beta$. Note the minus sign for the second term, which implies the trade-off between the internal energy and the entropy. 

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/posts_stats_mech_stats_inf/free_energy.png?raw=true" width="450rem">



This interactive relation in the free energy minimization argument reminds us of Bayesian statistics. In the absence of other information, our *prior* knowledge would say that $\mathbf{p}$ is uniform (for maximizing the entropy). On the other hand, with energy information, $\langle \epsilon \rangle_{\mathbf{p}}$ is minimized when $p$ is concentrated to a certain state. The interplay is controlled by $\beta$, a reciprocal of the temperature[^8].

# Variational free energy for inference

### Variational method in physics

Variational method[^5] is a technique for the approximation of complicated probability distributions. It has been used widely in physics. Take a simple example from quantum mechanics. Suppose the time-independent Schrodinger equation:

$$ H \psi = E \psi $$

where $H$ is a known time-independent Hamiltonian, $E$ is a energy, and $\psi$ is a wavefunction: The variational principle states that the ground-state energy, $E_0$, is always less than or equal to the expectation value of $H$ calculated with the trial wavefunction:

$$ E_0 \leq\langle \psi | H |\psi  \rangle$$

Therefore, by varying $\psi$ until the expectation value of $H$ is minimized, we can obtain approximations to the wavefunction and the energy of the ground-state.

<br>

### Variational method in statistics

Variational method also has been developed in statistics, for which I'll briefly describe here. Let $x$ as observations (data), $z$ as hidden variables, as $\theta$ as parameters. Suppose we're interested in the posterior distribution:

$$ p(z|x, \theta) = \cfrac{p(z,x|\theta)}{p(x|\theta)} =  \cfrac{p(z,x|\theta)}{\displaystyle\int_z p(z,x|\theta) dz} $$

But this quantity is often computationally intractable, mainly due to the denominator, since there is no closed form and the dimension to be considered is large. The key idea behind the variational method is to introduce an auxiliary distribution ($q$) over the hidden variable ($z$). Then, we find the parameters that make $q$ close to the posterior of interest. Here we measure the closeness of the two distributions with [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$ KL\big(q(z)\big|\big| p(z|x,\theta)\big) = E_q\Big[\log \cfrac{q(z)}{p(z | x, \theta)}\Big] $$

where $q$ resides in a distribution family $Q$. Because we are approximating the real but intractable distribution $p$ ($\in P$), by optimizing a simpler but tractable distribution $q$ from family $Q$. We are therefore varying the distribution from P to Q for the purpose of approximation.

<img src="https://github.com/uriyeobi/uriyeobi.github.io/blob/main/assets/images/posts_stats_mech_stats_inf/variational_method.png?raw=true" width="400rem">

One can see that we cannot minimize $KL$ directly, since we do not know $p(z \| x,\theta)$. But we can minimize a function that is equal to it up to a constant.

$$
\begin{aligned}
\log p(x|\theta) &= \log \displaystyle\int p(x,z|\theta) dz\\
&= \log \displaystyle\int q(z)\cfrac{p(x,z|\theta)}{q(z)} dz\\
&\geq \displaystyle\int q(z)\log \cfrac{p(x,z|\theta)}{q(z)} dz \ \ (\because \textsf{Jensen's inequality})\\
&=\displaystyle\int q(z)\log p(x,z|\theta) dz - \displaystyle\int q(z)\log q(z) dz \\
&=:L(q)
\end{aligned}
$$ 

where $L(q)$ is called [ELBO (evidence lower bound)](https://en.wikipedia.org/wiki/Evidence_lower_bound). By maximizing ELBO, we can increase a lower bound of $\log p(x\|\theta)$. 


### Variational inference

If we rewrite $L(q)$:

$$
\begin{aligned}
L(q)&=\displaystyle\int q(z)\log \cfrac{p(z|x,\theta)p(x|\theta)}{q(z)} dz\\
&=\displaystyle\int q(z)\log \cfrac{p(z|x,\theta)}{q(z)} dz + \log p(x|\theta) \displaystyle\int q(z) dz\\
&=-\displaystyle\int q(z)\log \cfrac{q(z)}{p(z|x,\theta)} dz + \log p(x|\theta)\\
&=-KL\big(q(z)\big|\big|p(z|x,\theta\big) + \log p(x|\theta)
\end{aligned}
$$

or,

$$\log p(x|\theta)=L(q) + KL\big(q(z)\big|\big|p(z|x,\theta\big) $$


Note that $\log p(x\|\theta)$ is independent from $q$, so a good approximation $q(z)$ of the exact Bayes' posterior will effectively minimize the KL divergence.

Note that $-L(q)$ is known as **Variational Free Energy**, since

$$
\begin{aligned}
-L(q) &= - \displaystyle\int q(z)\log p(x,z|\theta) dz + \displaystyle\int q(z)\log q(z) dz
\\ 
& =-E_q\Big[log(p(x,z)\Big] - H(q) 
\end{aligned}$$

**Variational inference** is a general framework to construct approximating probability distribution $q(z)$ to posterior distributions $p(z\|x)$ by minimizing functional

$$q^* = \underset{\mathbf{q}}{\mathrm{argmin}} \ \ KL\big(q(z)\big|\big|p(z|x,\theta)\big) = \underset{\mathbf{q}}{\mathrm{argmax}}\ \ L(q)$$

Thus, a variational free energy is an objective function used for variational inference. A well-known usage in physics would be [mean-field theory for Ising model](https://en.wikipedia.org/wiki/Mean-field_theory).

# Summary

Starting from enumerating particles, a well-known Boltzmann distribution was derived. Then we showed the parity between thermodynamics and information theory, by entropy maximum principle. The concept of free energy was reviewed from/to variational inference. All of these are not meant for full-fledged description of theory of each subject, but for demonstrating how one can understand statistical inference from statistical physics, and vice versa.

<br>


# References

- Bishop, C. M., Pattern Recognition and Machine Learning (chapter 10), Springer 2006. [[link]](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- Mackay, D. Information Theory, Inference, and Learning Algorithms (chapter 33), Cambridge Univ. Press: Cambridge, United Kingdom, 2003. [[link]](https://www.inference.org.uk/itprnn/book.pdf)
- Bahri Y., Kadmon J., Pennington J., Schoenholz S.S., Sohl-Dickstein J., Ganguli S., Statistical mechanics of deep learning, Annu. Rev. Condens. Matter Phys., 11 (2020), pp. 501-528. [[link]](https://www.annualreviews.org/doi/abs/10.1146/annurev-conmatphys-031119-05074)
- Blei, D. M., Kucukelbir, A. & McAuliffe, J. D., Variational inference: a review for statisticians. J. Am. Stat. Assoc. 112, 859–877 (2017). [[link]](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/BleiKucukelbirMcAuliffe2017.pdf) 
- Probabilistic Machine Learning (Summer 2020), Tübingen Machine Learning. [[link]](https://www.youtube.com/playlist?list=PL05umP7R6ij1tHaOFY96m5uX3J21a6yNd)
- Jaynes. E.T., Information theory and statistical mechanics, Phys. Rev. 106 (1957), p. 620. [[link]](https://bayes.wustl.edu/etj/articles/theory.1.pdf)
- Free Energy Principle by Karl Friston: Neuroscience and the Free Energy Principle, Lex Fridman Podcast. [[link]](https://youtu.be/NwzuibY5kUs?t=2590)
- Tutorial "Variational Bayes and beyond: Bayesian inference for big data", ICML 2018. [[link]](https://tamarabroderick.com/tutorial_2020_smiles.html)
- Beal, M. J. (2003), Variational algorithms for approximate Bayesian inference. PhD Thesis. University College London, London [[link]](https://cse.buffalo.edu/faculty/mbeal/papers/beal03.pdf)
- Zuckerman, D. M., Statistical Physics of Biomolecules: An Introduction. CRC Press; 2010. [[link]](https://www.taylorfrancis.com/books/mono/10.1201/b18849/statistical-physics-biomolecules-daniel-zuckerman)
- Gottwald, S. and Braun, D. A., The two kinds of free energy and the Bayesian revolution, PLoS Computational Biology, 16(12):e1008420, 2020. [[link]](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008420)
- Friston, K. The free-energy principle: a rough guide to the brain? Trends Cogn. Sci. 13, 293–301 (2009). [[link]](https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf)

****





**Notes**

[^1]: What is entropy? Entropy may rank as the worst-explained important idea in physics, chemistry, and biology. The concept was not clear in early history. Here is a quote from [John von Neumann](https://en.wikipedia.org/wiki/John_von_Neumann): *"You should call it entropy, for two reasons. In the first place your uncertainty function has been used in statistical mechanics under that name, so it already has a name. In the second place, and more important, no one really knows what entropy really is, so in a debate you will always have the advantage."*, from [Scientific American Vol. 225 No. 3, (1971), p. 180](http://ftp.math.utah.edu/pub/tex/bib/toc/sciam1970.html#224(3):March:1971).

[^2]: More generally, it is expressed in macroscopic variables[^2] as $ F = H - TS$, where $H$ enthalpy. Another definition by the partition function ($Z$) is: $F=-\cfrac{1}{\beta}\ln Z$, where $\beta=\cfrac{1}{k_BT}$ is the reciprocal temperature.

[^4]: such an ensemble is referred to as a [canonical ensemble](https://en.wikipedia.org/wiki/Canonical_ensemble).

[^5]: The historical origin of this method is the [calculus of variations](https://en.wikipedia.org/wiki/Calculus_of_variations)

[^7]: This formulation can be actually derived from the variational principle.

[^8]: Temperature plays an important role. For example, consider [Ising model](https://en.wikipedia.org/wiki/Ising_model) for [ferromagnet](https://en.wikipedia.org/wiki/Ferromagnetism). In the disordered paramagnetic phase (low temperature), the spins can point in any direction, so there are numerous possible configurations (low energy / high entropy), but there is little energetic reward, since the spins haven't aligned. In the ordered ferromagnetic phase (lower temperature), spins are mostly in one direction, so there are less possible configurations (high energy / low entropy). 