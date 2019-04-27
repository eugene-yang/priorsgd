# Stochastic Gradient Descent with Priors (priorsgd)
This implementation is based on `sklearn.linear_model.SGDClassifier`.
We modify the underlying basic stochastic gradient descent algorithm to support
1. Per feature L1 and L2 regularization
2. The target vector the algorithm will regularize the weight vector toward.

Part of the package is written in Cython, which is a mixture of Python and C.
This require compiling before use. The package is **still loosely depends on**
**the original scikit-learn package**. 

## Requriments
Recommend to use Anaconda for package and external environment control(including 
gcc and CBLAS). Please use the Python 3.6 version of Anaconda.
https://www.anaconda.com/download/#macos

- Python 3.6
- cython >= 0.27.3
- numpy >= 1.14.0
- scikit-learn = 0.19.1
- scipy >= 1.1.0

## Compile and Install
Please make sure `numpy` is already installed.
And simply run `pip install ./`.

## Key Modifications
The scripts are modified from the original scikit-learn library.
Detail modification can be found by using git diff between commit
`5fcf6f486fb0134c91f5a8741fe7e450539883f0` in original scikit-learn 
[repository](https://github.com/scikit-learn/scikit-learn).

1. (From `sklearn/linear_model/stochastic_gradient.py`) 
The `stochastic_gradient.py` provides the python interface. 
The modifications are the additional arguments in `SGDClassifier.fit()`.
The following are the added arguemnts. Detail explanation are in the 
documentation of the arugment in the code. 
    - `per_feature_alpha`(per feature L2 penalty)
    - `per_feature_beta`(per feature L1 penalty)
    - `modal_vector`(prior vector)

2. (From `sklearn/linear_model/sgd_fast.pyx`) 
The main algorithm is in `sgd_fast.pyx`, this script will further compile to 
C binary file by Cython. The `plain_sgd` function provides the interface 
between python and C for the entire algorithm.

3. (From `sklearn/util/weight_vector.pyx`) 
The applying of the regularization is hidden in the method `apply_penalty`  
in `weight_vector.pyx`. This method would apply both L1 and L2 penalty 
**without** using any efficiency tricks, such as JIT update or delayed update.

4. (From `sklearn/util/seq_dataset.pyx`) 
`seq_dataset.pyx` is added to the directory for compiling purpose. 
This script is not modified. 

## Citations
Please kindly cite our paper if you are using this package.

If you are using non-zero priors(`modal_vector`), please cite:
```bibtex
@inproceedings{yang_regularization_2019,
	address = {Montréal, Canada},
	title = {A {Regularization} {Approach} to {Combining} {Keywords} and {Training} {Data} in {Technology}-{Assisted} {Review}},
	booktitle = {Proceedings of the 17th edition of the {International} {Conference} on {Artificial} {Intelligence} and {Law} ({ICAIL})},
	author = {Yang, Eugene and Lewis, David D. and Frieder, Ophir},
	month = jun,
	year = {2019}
}
```

If you are also using per feature regularization penalty(`per_feature_alpha` or
`per_feature_beta`), please cite:
```bibtex
@inproceedings{yang_text_2019,
	address = {Paris, France},
	title = {Text {Retrieval} {Priors} for {Bayesian} {Logistic} {Regression}},
	booktitle = {Proceedings of the 41st {International} {ACM} {SIGIR} {Conference} on {Research} and {Development} in {Information} {Retrieval}},
	publisher = {ACM},
	author = {Yang, Eugene and Lewis, David D. and Frieder, Ophir},
	month = jul,
	year = {2019}
}
```