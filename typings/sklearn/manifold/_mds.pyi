"""
This type stub file was generated by pyright.
"""

from numbers import Integral, Real
from ..base import BaseEstimator, _fit_context
from ..utils._param_validation import Interval, StrOptions, validate_params

"""
Multi-dimensional Scaling (MDS).
"""
@validate_params({ "dissimilarities": ["array-like"],"metric": ["boolean"],"n_components": [Interval(Integral, 1, None, closed="left")],"init": ["array-like", None],"n_init": [Interval(Integral, 1, None, closed="left")],"n_jobs": [Integral, None],"max_iter": [Interval(Integral, 1, None, closed="left")],"verbose": ["verbose"],"eps": [Interval(Real, 0, None, closed="left")],"random_state": ["random_state"],"return_n_iter": ["boolean"],"normalized_stress": ["boolean", StrOptions("auto")] }, prefer_skip_nested_validation=True)
def smacof(dissimilarities, *, metric=..., n_components=..., init=..., n_init=..., n_jobs=..., max_iter=..., verbose=..., eps=..., random_state=..., return_n_iter=..., normalized_stress=...): # -> tuple[Any | None, Any | None, Any] | tuple[Any | None, Any | None]:
    """Compute multidimensional scaling using the SMACOF algorithm.

    The SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm is a
    multidimensional scaling algorithm which minimizes an objective function
    (the *stress*) using a majorization technique. Stress majorization, also
    known as the Guttman Transform, guarantees a monotone convergence of
    stress, and is more powerful than traditional techniques such as gradient
    descent.

    The SMACOF algorithm for metric MDS can be summarized by the following
    steps:

    1. Set an initial start configuration, randomly or not.
    2. Compute the stress
    3. Compute the Guttman Transform
    4. Iterate 2 and 3 until convergence.

    The nonmetric algorithm adds a monotonic regression step before computing
    the stress.

    Parameters
    ----------
    dissimilarities : array-like of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Must be symmetric.

    metric : bool, default=True
        Compute metric or nonmetric SMACOF algorithm.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities. If an
        ``init`` array is provided, this option is overridden and the shape of
        ``init`` is used to determine the dimensionality of the embedding
        space.

    init : array-like of shape (n_samples, n_components), default=None
        Starting configuration of the embedding to initialize the algorithm. By
        default, the algorithm is initialized with a randomly chosen array.

    n_init : int, default=8
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress. If ``init`` is
        provided, this option is overridden and a single run is performed.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    normalized_stress : bool or "auto" default="auto"
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

        .. versionadded:: 1.2

        .. versionchanged:: 1.4
           The default value changed from `False` to `"auto"` in version 1.4.

    Returns
    -------
    X : ndarray of shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.

    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    n_iter : int
        The number of iterations corresponding to the best stress. Returned
        only if ``return_n_iter`` is set to ``True``.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
           Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
           hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
           Groenen P. Springer Series in Statistics (1997)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import smacof
    >>> from sklearn.metrics import euclidean_distances
    >>> X = np.array([[0, 1, 2], [1, 0, 3],[2, 3, 0]])
    >>> dissimilarities = euclidean_distances(X)
    >>> mds_result, stress = smacof(dissimilarities, n_components=2, random_state=42)
    >>> mds_result
    array([[ 0.05... -1.07... ],
           [ 1.74..., -0.75...],
           [-1.79...,  1.83...]])
    >>> stress
    np.float64(0.0012...)
    """
    ...

class MDS(BaseEstimator):
    """Multidimensional scaling.

    Read more in the :ref:`User Guide <multidimensional_scaling>`.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.

    metric : bool, default=True
        If ``True``, perform metric MDS; otherwise, perform nonmetric MDS.
        When ``False`` (i.e. non-metric MDS), dissimilarities with 0 are considered as
        missing values.

    n_init : int, default=4
        Number of times the SMACOF algorithm will be run with different
        initializations. The final results will be the best output of the runs,
        determined by the run with the smallest final stress.

    max_iter : int, default=300
        Maximum number of iterations of the SMACOF algorithm for a single run.

    verbose : int, default=0
        Level of verbosity.

    eps : float, default=1e-3
        Relative tolerance with respect to stress at which to declare
        convergence. The value of `eps` should be tuned separately depending
        on whether or not `normalized_stress` is being used.

    n_jobs : int, default=None
        The number of jobs to use for the computation. If multiple
        initializations are used (``n_init``), each run of the algorithm is
        computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    normalized_stress : bool or "auto" default="auto"
        Whether use and return normed stress value (Stress-1) instead of raw
        stress calculated by default. Only supported in non-metric MDS.

        .. versionadded:: 1.2

        .. versionchanged:: 1.4
           The default value changed from `False` to `"auto"` in version 1.4.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Stores the position of the dataset in the embedding space.

    stress_ : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
        If `normalized_stress=True`, and `metric=False` returns Stress-1.
        A value of 0 indicates "perfect" fit, 0.025 excellent, 0.05 good,
        0.1 fair, and 0.2 poor [1]_.

    dissimilarity_matrix_ : ndarray of shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Symmetric matrix that:

        - either uses a custom dissimilarity matrix by setting `dissimilarity`
          to 'precomputed';
        - or constructs a dissimilarity matrix from data using
          Euclidean distances.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_iter_ : int
        The number of iterations corresponding to the best stress.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    TSNE : T-distributed Stochastic Neighbor Embedding.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    References
    ----------
    .. [1] "Nonmetric multidimensional scaling: a numerical method" Kruskal, J.
       Psychometrika, 29 (1964)

    .. [2] "Multidimensional scaling by optimizing goodness of fit to a nonmetric
       hypothesis" Kruskal, J. Psychometrika, 29, (1964)

    .. [3] "Modern Multidimensional Scaling - Theory and Applications" Borg, I.;
       Groenen P. Springer Series in Statistics (1997)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import MDS
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = MDS(n_components=2, normalized_stress='auto')
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_manifold_plot_mds.py`.

    For a comparison of manifold learning techniques, see
    :ref:`sphx_glr_auto_examples_manifold_plot_compare_methods.py`.
    """
    _parameter_constraints: dict = ...
    def __init__(self, n_components=..., *, metric=..., n_init=..., max_iter=..., verbose=..., eps=..., n_jobs=..., random_state=..., dissimilarity=..., normalized_stress=...) -> None:
        ...
    
    def __sklearn_tags__(self): # -> Tags:
        ...
    
    def fit(self, X, y=..., init=...): # -> Self:
        """
        Compute the position of the points in the embedding space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        ...
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=..., init=...): # -> Any | None:
        """
        Fit the data from `X`, and returns the embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.

        y : Ignored
            Not used, present for API consistency by convention.

        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            X transformed in the new space.
        """
        ...
    


