"""
This type stub file was generated by pyright.
"""

from abc import ABCMeta, abstractmethod
from numbers import Real
from .base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from .utils._param_validation import Interval, validate_params

"""Random projection transformers.

Random projections are a simple and computationally efficient way to
reduce the dimensionality of the data by trading a controlled amount
of accuracy (as additional variance) for faster processing times and
smaller model sizes.

The dimensions and distribution of random projections matrices are
controlled so as to preserve the pairwise distances between any two
samples of the dataset.

The main theoretical result behind the efficiency of random projection is the
`Johnson-Lindenstrauss lemma (quoting Wikipedia)
<https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma>`_:

  In mathematics, the Johnson-Lindenstrauss lemma is a result
  concerning low-distortion embeddings of points from high-dimensional
  into low-dimensional Euclidean space. The lemma states that a small set
  of points in a high-dimensional space can be embedded into a space of
  much lower dimension in such a way that distances between the points are
  nearly preserved. The map used for the embedding is at least Lipschitz,
  and can even be taken to be an orthogonal projection.
"""
__all__ = ["SparseRandomProjection", "GaussianRandomProjection", "johnson_lindenstrauss_min_dim"]
@validate_params({ "n_samples": ["array-like", Interval(Real, 1, None, closed="left")],"eps": ["array-like", Interval(Real, 0, 1, closed="neither")] }, prefer_skip_nested_validation=True)
def johnson_lindenstrauss_min_dim(n_samples, *, eps=...): # -> ndarray[_Shape, dtype[signedinteger[_64Bit]]]:
    """Find a 'safe' number of components to randomly project to.

    The distortion introduced by a random projection `p` only changes the
    distance between two points by a factor (1 +- eps) in a euclidean space
    with good probability. The projection `p` is an eps-embedding as defined
    by:

    .. code-block:: text

      (1 - eps) ||u - v||^2 < ||p(u) - p(v)||^2 < (1 + eps) ||u - v||^2

    Where u and v are any rows taken from a dataset of shape (n_samples,
    n_features), eps is in ]0, 1[ and p is a projection by a random Gaussian
    N(0, 1) matrix of shape (n_components, n_features) (or a sparse
    Achlioptas matrix).

    The minimum number of components to guarantee the eps-embedding is
    given by:

    .. code-block:: text

      n_components >= 4 log(n_samples) / (eps^2 / 2 - eps^3 / 3)

    Note that the number of dimensions is independent of the original
    number of features but instead depends on the size of the dataset:
    the larger the dataset, the higher is the minimal dimensionality of
    an eps-embedding.

    Read more in the :ref:`User Guide <johnson_lindenstrauss>`.

    Parameters
    ----------
    n_samples : int or array-like of int
        Number of samples that should be an integer greater than 0. If an array
        is given, it will compute a safe number of components array-wise.

    eps : float or array-like of shape (n_components,), dtype=float, \
            default=0.1
        Maximum distortion rate in the range (0, 1) as defined by the
        Johnson-Lindenstrauss lemma. If an array is given, it will compute a
        safe number of components array-wise.

    Returns
    -------
    n_components : int or ndarray of int
        The minimal number of components to guarantee with good probability
        an eps-embedding with n_samples.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

    .. [2] `Sanjoy Dasgupta and Anupam Gupta, 1999,
           "An elementary proof of the Johnson-Lindenstrauss Lemma."
           <https://citeseerx.ist.psu.edu/doc_view/pid/95cd464d27c25c9c8690b378b894d337cdf021f9>`_

    Examples
    --------
    >>> from sklearn.random_projection import johnson_lindenstrauss_min_dim
    >>> johnson_lindenstrauss_min_dim(1e6, eps=0.5)
    np.int64(663)

    >>> johnson_lindenstrauss_min_dim(1e6, eps=[0.5, 0.1, 0.01])
    array([    663,   11841, 1112658])

    >>> johnson_lindenstrauss_min_dim([1e4, 1e5, 1e6], eps=0.1)
    array([ 7894,  9868, 11841])
    """
    ...

class BaseRandomProjection(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for random projections.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    _parameter_constraints: dict = ...
    @abstractmethod
    def __init__(self, n_components=..., *, eps=..., compute_inverse_components=..., random_state=...) -> None:
        ...
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=...): # -> Self:
        """Generate a sparse random projection matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training set: only the shape is used to find optimal random
            matrix dimensions based on the theory referenced in the
            afore mentioned papers.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            BaseRandomProjection class instance.
        """
        ...
    
    def inverse_transform(self, X):
        """Project data back to its original space.

        Returns an array X_original whose transform would be X. Note that even
        if X is sparse, X_original is dense: this may use a lot of RAM.

        If `compute_inverse_components` is False, the inverse of the components is
        computed during each call to `inverse_transform` which can be costly.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_components)
            Data to be transformed back.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        ...
    
    def __sklearn_tags__(self):
        ...
    


class GaussianRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through Gaussian random projection.

    The components of the random matrix are drawn from N(0, 1 / n_components).

    Read more in the :ref:`User Guide <gaussian_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when `n_components` is set to
        'auto'. The value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    compute_inverse_components : bool, default=False
        Learn the inverse transform by computing the pseudo-inverse of the
        components during fit. Note that computing the pseudo-inverse does not
        scale well to large matrices.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : ndarray of shape (n_components, n_features)
        Random matrix used for the projection.

    inverse_components_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of the components, only computed if
        `compute_inverse_components` is True.

        .. versionadded:: 1.1

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SparseRandomProjection : Reduce dimensionality through sparse
        random projection.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import GaussianRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = GaussianRandomProjection(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    """
    def __init__(self, n_components=..., *, eps=..., compute_inverse_components=..., random_state=...) -> None:
        ...
    
    def transform(self, X):
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected array.
        """
        ...
    


class SparseRandomProjection(BaseRandomProjection):
    """Reduce dimensionality through sparse random projection.

    Sparse random matrix is an alternative to dense random
    projection matrix that guarantees similar embedding quality while being
    much more memory efficient and allowing faster computation of the
    projected data.

    If we note `s = 1 / density` the components of the random matrix are
    drawn from:

    .. code-block:: text

      -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
       0                              with probability 1 - 1 / s
      +sqrt(s) / sqrt(n_components)   with probability 1 / 2s

    Read more in the :ref:`User Guide <sparse_random_matrix>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_components : int or 'auto', default='auto'
        Dimensionality of the target projection space.

        n_components can be automatically adjusted according to the
        number of samples in the dataset and the bound given by the
        Johnson-Lindenstrauss lemma. In that case the quality of the
        embedding is controlled by the ``eps`` parameter.

        It should be noted that Johnson-Lindenstrauss lemma can yield
        very conservative estimated of the required number of components
        as it makes no assumption on the structure of the dataset.

    density : float or 'auto', default='auto'
        Ratio in the range (0, 1] of non-zero component in the random
        projection matrix.

        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).

        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.

    eps : float, default=0.1
        Parameter to control the quality of the embedding according to
        the Johnson-Lindenstrauss lemma when n_components is set to
        'auto'. This value should be strictly positive.

        Smaller values lead to better embedding and higher number of
        dimensions (n_components) in the target projection space.

    dense_output : bool, default=False
        If True, ensure that the output of the random projection is a
        dense numpy array even if the input and random projection matrix
        are both sparse. In practice, if the number of components is
        small the number of zero components in the projected data will
        be very small and it will be more CPU and memory efficient to
        use a dense representation.

        If False, the projected data uses a sparse representation if
        the input is sparse.

    compute_inverse_components : bool, default=False
        Learn the inverse transform by computing the pseudo-inverse of the
        components during fit. Note that the pseudo-inverse is always a dense
        array, even if the training data was sparse. This means that it might be
        necessary to call `inverse_transform` on a small batch of samples at a
        time to avoid exhausting the available memory on the host. Moreover,
        computing the pseudo-inverse does not scale well to large matrices.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the
        projection matrix at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_components_ : int
        Concrete number of components computed when n_components="auto".

    components_ : sparse matrix of shape (n_components, n_features)
        Random matrix used for the projection. Sparse matrix will be of CSR
        format.

    inverse_components_ : ndarray of shape (n_features, n_components)
        Pseudo-inverse of the components, only computed if
        `compute_inverse_components` is True.

        .. versionadded:: 1.1

    density_ : float in range 0.0 - 1.0
        Concrete density computed from when density = "auto".

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    GaussianRandomProjection : Reduce dimensionality through Gaussian
        random projection.

    References
    ----------

    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf

    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           https://cgi.di.uoa.gr/~optas/papers/jl.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.random_projection import SparseRandomProjection
    >>> rng = np.random.RandomState(42)
    >>> X = rng.rand(25, 3000)
    >>> transformer = SparseRandomProjection(random_state=rng)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (25, 2759)
    >>> # very few components are non-zero
    >>> np.mean(transformer.components_ != 0)
    np.float64(0.0182...)
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_components=..., *, density=..., eps=..., dense_output=..., compute_inverse_components=..., random_state=...) -> None:
        ...
    
    def transform(self, X): # -> Any:
        """Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Projected array. It is a sparse matrix only when the input is sparse and
            `dense_output = False`.
        """
        ...
    


