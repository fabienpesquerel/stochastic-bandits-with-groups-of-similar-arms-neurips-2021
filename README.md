# Experiments
## How to reproduce experimental results of *Stochastic bandits with groups of similar arms* submitted paper ?

### Section 5 of the paper
To reproduce all of the empirical results (and more) of the section 5 (Experiments) of the submitted paper, it is enough to run the `regret_experiment.ipynb` notebook.

### Appendix D of the paper
To reproduce all of the empirical results (and more) of the appendix D (Experiments) of the submitted paper, you may run the `regret_experiment.ipynb` notebook as well as the `dispatching_experiment.ipynb` notebook.

### How to?
To run the jupyter notebooks, you may use the jupyter software (https://jupyter.org/) or use an online plateform such as google Colab (https://colab.research.google.com/)

### Which versions
The code has run on google Colab on Friday 4th, June without problem. The stability of the package that were used (numpy, matplotlib, itertools (built-in), datetime (built-in)) makes it very likely to run correctly before the next major updates of one of those packages.

Otherwise, the code has been tested locally on a machine with the following versions :
python 3.9.5
numpy 1.20.3
matplotlib 3.4.1

### Where to look at?
The `algorithms.py` file contains all the algorithms that were used in the experiments of this paper. All the sequential algorithms depends on `SequentiAlg`, a class that is defined in the `Forban` module.

In the `Forban` module, a class for `Bandit` configurations is also defined.