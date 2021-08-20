# Heritability Maximizing Dimensionality Reduction

## The Model
- N individuals, indexed by i
- M variants, indexed by j
- K phenotypes, indexed by k

- h<sup>2</sup> = 0.5

- p<sub>j</sub> ~ Uniform(0.05, 0.95)
- X<sub>ij</sub> ~ Binomial(2, p)
- β<sub>jk</sub> ~ Normal(0, h<sup>2</sup> / M)
- 𝓁<sub>ik</sub> ~ X<sub>ij</sub> · β<sub>jk</sub> + Normal(0, √(1-h<sup>2</sup>))
- y<sub>ik</sub> = f(𝓁<sub>ik</sub>), where f is some non-linear function

## Next Steps
- [ ] add noise to the images before training the model
- [ ] understand why the variance of trace-heritability after linear transformations is not zero;
  understand how this is related to variance explained between estimated and simulated latent
  phenotypes.
- [ ] scale up test setup (use more computers? speed up Balding-Nichols? ???)
- [ ] handle variants with non-zero LD
- [ ] handle samples with non-zero relatedness
