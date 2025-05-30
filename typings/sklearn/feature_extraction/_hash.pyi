"""
This type stub file was generated by pyright.
"""

from ..base import BaseEstimator, TransformerMixin, _fit_context

class FeatureHasher(TransformerMixin, BaseEstimator):
    """Implements feature hashing, aka the hashing trick.

    This class turns sequences of symbolic feature names (strings) into
    scipy.sparse matrices, using a hash function to compute the matrix column
    corresponding to a name. The hash function employed is the signed 32-bit
    version of Murmurhash3.

    Feature names of type byte string are used as-is. Unicode strings are
    converted to UTF-8 first, but no Unicode normalization is done.
    Feature values must be (finite) numbers.

    This class is a low-memory alternative to DictVectorizer and
    CountVectorizer, intended for large-scale (online) learning and situations
    where memory is tight, e.g. when running prediction code on embedded
    devices.

    For an efficiency comparison of the different feature extractors, see
    :ref:`sphx_glr_auto_examples_text_plot_hashing_vs_dict_vectorizer.py`.

    Read more in the :ref:`User Guide <feature_hashing>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    n_features : int, default=2**20
        The number of features (columns) in the output matrices. Small numbers
        of features are likely to cause hash collisions, but large numbers
        will cause larger coefficient dimensions in linear learners.
    input_type : str, default='dict'
        Choose a string from {'dict', 'pair', 'string'}.
        Either "dict" (the default) to accept dictionaries over
        (feature_name, value); "pair" to accept pairs of (feature_name, value);
        or "string" to accept single strings.
        feature_name should be a string, while value should be a number.
        In the case of "string", a value of 1 is implied.
        The feature_name is hashed to find the appropriate column for the
        feature. The value's sign might be flipped in the output (but see
        non_negative, below).
    dtype : numpy dtype, default=np.float64
        The type of feature values. Passed to scipy.sparse matrix constructors
        as the dtype argument. Do not set this to bool, np.boolean or any
        unsigned integer type.
    alternate_sign : bool, default=True
        When True, an alternating sign is added to the features as to
        approximately conserve the inner product in the hashed space even for
        small n_features. This approach is similar to sparse random projection.

        .. versionchanged:: 0.19
            ``alternate_sign`` replaces the now deprecated ``non_negative``
            parameter.

    See Also
    --------
    DictVectorizer : Vectorizes string-valued features using a hash table.
    sklearn.preprocessing.OneHotEncoder : Handles nominal/categorical features.

    Notes
    -----
    This estimator is :term:`stateless` and does not need to be fitted.
    However, we recommend to call :meth:`fit_transform` instead of
    :meth:`transform`, as parameter validation is only performed in
    :meth:`fit`.

    Examples
    --------
    >>> from sklearn.feature_extraction import FeatureHasher
    >>> h = FeatureHasher(n_features=10)
    >>> D = [{'dog': 1, 'cat':2, 'elephant':4},{'dog': 2, 'run': 5}]
    >>> f = h.transform(D)
    >>> f.toarray()
    array([[ 0.,  0., -4., -1.,  0.,  0.,  0.,  0.,  0.,  2.],
           [ 0.,  0.,  0., -2., -5.,  0.,  0.,  0.,  0.,  0.]])

    With `input_type="string"`, the input must be an iterable over iterables of
    strings:

    >>> h = FeatureHasher(n_features=8, input_type="string")
    >>> raw_X = [["dog", "cat", "snake"], ["snake", "dog"], ["cat", "bird"]]
    >>> f = h.transform(raw_X)
    >>> f.toarray()
    array([[ 0.,  0.,  0., -1.,  0., -1.,  0.,  1.],
           [ 0.,  0.,  0., -1.,  0., -1.,  0.,  0.],
           [ 0., -1.,  0.,  0.,  0.,  0.,  0.,  1.]])
    """
    __metadata_request__transform = ...
    _parameter_constraints: dict = ...
    def __init__(self, n_features=..., *, input_type=..., dtype=..., alternate_sign=...) -> None:
        ...
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X=..., y=...): # -> Self:
        """Only validates estimator's parameters.

        This method allows to: (i) validate the estimator's parameters and
        (ii) be consistent with the scikit-learn transformer API.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            FeatureHasher class instance.
        """
        ...
    
    def transform(self, raw_X): # -> csr_matrix:
        """Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        raw_X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        """
        ...
    
    def __sklearn_tags__(self):
        ...
    


