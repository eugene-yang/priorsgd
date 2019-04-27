"""
The :mod:`sklearn.linear_model` module implements generalized linear models. It
includes Ridge regression, Bayesian Regression, Lasso and Elastic Net
estimators computed with Least Angle Regression and coordinate descent. It also
implements Stochastic Gradient Descent related algorithms.
"""

# See http://scikit-learn.sourceforge.net/modules/sgd.html and
# http://scikit-learn.sourceforge.net/modules/linear_model.html for
# complete documentation.


from .sgd_fast import Hinge, Log, ModifiedHuber, SquaredLoss, Huber
from .stochastic_gradient import SGDClassifier

__all__ = ['Hinge',
           'Huber',
           'Log',
           'ModifiedHuber',
           'SGDClassifier',
           'SquaredLoss'
           ]
