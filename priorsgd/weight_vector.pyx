# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
#         Danny Sullivan <dsullivan7@hotmail.com>
#
# License: BSD 3 clause

cimport cython
from libc.limits cimport INT_MAX
from libc.math cimport sqrt, pow, fabs
import numpy as np
cimport numpy as np
ctypedef np.uint8_t uint8

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil
    void daxpy "cblas_daxpy" (int, double, const double*,
                              int, double*, int) nogil

np.import_array()


cdef class WeightVector(object):
    """Dense vector represented by a scalar and a numpy array.

    The class provides methods to ``add`` a sparse vector
    and scale the vector.
    Representing a vector explicitly as a scalar times a
    vector allows for efficient scaling operations.

    Attributes
    ----------
    w : ndarray, dtype=double, order='C'
        The numpy array which backs the weight vector.
    aw : ndarray, dtype=double, order='C'
        The numpy array which backs the average_weight vector.
    w_data_ptr : double*
        A pointer to the data of the numpy array.
    wscale : double
        The scale of the vector.
    n_features : int
        The number of features (= dimensionality of ``w``).
    sq_norm : double
        The squared norm of ``w``.
    """

    def __cinit__(self,
                  np.ndarray[double, ndim=1, mode='c'] w,
                  np.ndarray[double, ndim=1, mode='c'] aw,
                  np.ndarray[double, ndim=1, mode='c'] pf_alpha,
                  np.ndarray[double, ndim=1, mode='c'] pf_beta,
                  np.ndarray[double, ndim=1, mode='c'] modal,
                  int n_samples):
        cdef double *wdata = <double *>w.data

        if w.shape[0] > INT_MAX:
            raise ValueError("More than %d features not supported; got %d."
                             % (INT_MAX, w.shape[0]))
        self.w = w
        self.w_data_ptr = wdata
        self.wscale = 1.0
        self.n_features = w.shape[0]
        self.n_samples = n_samples

        # depreciated: will not be maintaining the norm /correctly/
        self.sq_norm = ddot(<int>w.shape[0], wdata, 1, wdata, 1)

        self.aw = aw
        if self.aw is not None:
            self.aw_data_ptr = <double *>aw.data
            self.average_a = 0.0
            self.average_b = 1.0
        
        self.pf_alpha_ptr = &pf_alpha[0]
        self.pf_beta_ptr = &pf_beta[0]
        self.modal_ptr = &modal[0]
    
    cdef void clear(self) nogil:
        cdef int j

        cdef double* w_data_ptr = self.w_data_ptr
        cdef int n_features = self.n_features

        for j in range(n_features):
            w_data_ptr[j] = 0


    cdef void add(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                  double c) nogil:
        """Scales sample x by constant c and adds it to the weight vector.

        This operation updates ``sq_norm``.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double innerprod = 0.0
        cdef double xsqnorm = 0.0

        # the next two lines save a factor of 2!
        cdef double wscale = self.wscale
        cdef double* w_data_ptr = self.w_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            innerprod += (w_data_ptr[idx] * val)
            xsqnorm += (val * val)
            w_data_ptr[idx] += val * (c / wscale)

        self.sq_norm += (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

    # Update the average weights according to the sparse trick defined
    # here: http://research.microsoft.com/pubs/192769/tricks-2012.pdf
    # by Leon Bottou
    cdef void add_average(self, double *x_data_ptr, int *x_ind_ptr, int xnnz,
                          double c, double num_iter) nogil:
        """Updates the average weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x``.
        c : double
            The scaling constant for the example.
        num_iter : double
            The total number of iterations.
        """
        cdef int j
        cdef int idx
        cdef double val
        cdef double mu = 1.0 / num_iter
        cdef double average_a = self.average_a
        cdef double wscale = self.wscale
        cdef double* aw_data_ptr = self.aw_data_ptr

        for j in range(xnnz):
            idx = x_ind_ptr[j]
            val = x_data_ptr[j]
            aw_data_ptr[idx] += (self.average_a * val * (-c / wscale))

        # Once the sample has been processed
        # update the average_a and average_b
        if num_iter > 1:
            self.average_b /= (1.0 - mu)
        self.average_a += mu * self.average_b * wscale

    cdef double dot(self, double *x_data_ptr, int *x_ind_ptr,
                    int xnnz) nogil:
        """Computes the dot product of a sample x and the weight vector.

        Parameters
        ----------
        x_data_ptr : double*
            The array which holds the feature values of ``x``.
        x_ind_ptr : np.intc*
            The array which holds the feature indices of ``x``.
        xnnz : int
            The number of non-zero features of ``x`` (length of x_ind_ptr).

        Returns
        -------
        innerprod : double
            The inner product of ``x`` and ``w``.
        """
        cdef int j
        cdef int idx
        cdef double innerprod = 0.0
        cdef double* w_data_ptr = self.w_data_ptr
        for j in range(xnnz):
            idx = x_ind_ptr[j]
            innerprod += w_data_ptr[idx] * x_data_ptr[j]
        innerprod *= self.wscale
        return innerprod

    cdef void scale(self, double c) nogil:
        """Scales the weight vector by a constant ``c``.

        It updates ``wscale`` and ``sq_norm``. If ``wscale`` gets too
        small we call ``reset_swcale``."""
        self.wscale *= c
        self.sq_norm *= (c * c)
        if self.wscale < 1e-9:
            self.reset_wscale()
    
    cdef void apply_penalty(self, double eta,
                            int *x_ind_ptr, int xnnz
                            ) nogil:
        cdef int i, idx
        cdef int n_features = self.n_features
        cdef double rate, effective_w
        cdef double l2update
        cdef double* w_data_ptr = self.w_data_ptr
        cdef double* pf_alpha_ptr = self.pf_alpha_ptr
        cdef double* pf_beta_ptr = self.pf_beta_ptr
        cdef double* modal_ptr = self.modal_ptr

        for idx in range(n_features):
            
            
            l2update = 2.0 * pf_alpha_ptr[idx] * eta * ( w_data_ptr[idx] - modal_ptr[idx] )
            if fabs(w_data_ptr[idx]) < fabs(l2update) and w_data_ptr[idx] * l2update > 0:
                w_data_ptr[idx] = 0.0
            else:    
                w_data_ptr[idx] = w_data_ptr[idx] - l2update

            if w_data_ptr[idx] - modal_ptr[idx] > pf_beta_ptr[idx] * eta :
                w_data_ptr[idx] = w_data_ptr[idx] - pf_beta_ptr[idx] * eta
            elif w_data_ptr[idx] - modal_ptr[idx] < -pf_beta_ptr[idx] * eta :
                w_data_ptr[idx] = w_data_ptr[idx] + pf_beta_ptr[idx] * eta
            else:  # in the middle
                w_data_ptr[idx] = modal_ptr[idx]

    cdef int copyto(self, double* backup_ptr, int current_idx) nogil:
        cdef int i
        cdef int n_features = self.n_features
        cdef double* w_data_ptr = self.w_data_ptr

        for i in range(n_features):
            backup_ptr[ current_idx + i ] = w_data_ptr[i]

        return current_idx + n_features

    cdef void reset_wscale(self) nogil:
        """Scales each coef of ``w`` by ``wscale`` and resets it to 1. """
        if self.aw is not None:
            daxpy(<int>self.aw.shape[0], self.average_a,
                  <double *>self.w.data, 1, <double *>self.aw.data, 1)
            dscal(<int>self.aw.shape[0], 1.0 / self.average_b,
                  <double *>self.aw.data, 1)
            self.average_a = 0.0
            self.average_b = 1.0

        dscal(<int>self.w.shape[0], self.wscale, <double *>self.w.data, 1)
        self.wscale = 1.0

    cdef double norm(self) nogil:
        """The L2 norm of the weight vector. """
        return sqrt(self.sq_norm)
