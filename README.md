# locally_efficient_differential_privacy

Consider the problem of publishing survey data without revealing sensitive information about its participants. Perhaps the most prevalent method of protecting private data is differential privacy. It introduces randomness to the collected datapoints, obfuscating individual information while retaining global properties of the overall dataset.

Formally, differential privacy is a Markov kernel $\kappa$ that maps $(X, A)$ measurable space to $(Z, B)$, $\kappa : (X, A) \to (Z, B)$, sometimes written as $\kappa : B \times X \to [0,1]$ with the following properties:

1. For every $B_0 \in B$, the map $x \mapsto \kappa(B_0, x)$ is $A$ measurable
2. For every $x_0 \in X$, the map $B \mapsto \kappa(B, x_0)$ is a probability measure on $(Y, B).

