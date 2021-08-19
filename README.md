# Heritability Maximizing Dimensionality Reduction

## The Model
- N individuals, indexed by i
- M variants, indexed by j
- K phenotypes, indexed by k

- h<sup>2</sup> = 0.5

- p<sub>j</sub> ~ Uniform(0.05, 0.95)
- X<sub>ij</sub> ~ Binomial(2, p)
- β<sub>jk</sub> ~ Normal(0, h<sup>2</sup> / M)
- 𝓁<sub>k</sub> ~ X<sub>ij</sub> · β<sub>jk</sub> + Normal(0, √(1-h<sup>2</sup>))
- y<sub>k</sub> = f(𝓁<sub>k</sub>), where f is some non-linear function
