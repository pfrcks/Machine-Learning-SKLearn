Numpy:
	- ones : Return a new array of given shape and type, filled with ones.
	- arange : Return evenly spaced values within a given interval. arange([start,] stop[, step,], dtype=None)
	- asarray : Convert the input to an array.
	- random.random : Return random floats in the half-open interval [0.0, 1.0).
	- linspace : Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`].
	- newaxis : Starting with 1D list of numbers with newaxis, you can turn it into a 2D matrix.
	- array : Create an array
	- random.normal : Draw random samples from a normal (Gaussian) distribution.
	- meshgrid : The purpose of meshgrid is to create a rectangular grid out of an array of x values and an array of y values.
	- random.RandomState : Container for the Mersenne Twister pseudo-random number generator
	- random.RandomState.permutation : Randomly permute a sequence, or return a permuted range
	- random.RandomState.uniform : Draw samples from a uniform distribution. 
	- squeeze : Remove single-dimensional entries from the shape of an array
	- random.randn : Return a sample (or samples) from the "standard normal" distribution.
	- dot : Dot product of two arrays.
	- ravel : Return a contiguous flattened array.
	- set_printoptions : Set printing options.

Matplotlib:
	- scatter : Make a scatter plot of x vs y, where x and y are sequence like objects of the same lengths.
	- contour : Makes a contour. 
	- figure : Creates a new figure. 
	- add_subplot : Add a subplot.
	- subplots_adjust : As implied by name.
	- pcolormesh : Create a pseudocolor plot of a 2-D array.
	- xlim : Get or set the *x* limits of the current axes.
	- ylim : Get or set the *y* limits of the current axes.
	- axis : Convenience method to get or set axis properties.
	- fill_between : Make filled polygons between two curves.
	- setp : Set a property on an artist object.

Scikit-Learn:
	- K Neighbors (Classifier/Regressor) : Classifier/Regressor implementing the k-nearest neighbors vote
	- linear_model : The :mod:`sklearn.linear_model` module implements generalized linear models. Eg SGD, BR, etc.
	- Linear Regression : Ordinary least squares Linear Regression.
	- ^ normalize? : If True, the regressors X will be normalized before regression.
	- coef_ : Estimated coefficients for the linear regression problem
	- intercept_ : Independent term in the linear model.
	- residues_ : Get the residues of the fitted model.
	- makeblobs : Generate isotropic Gaussian blobs for clustering.
	- make_circles : Make a large circle containing a smaller circle in 2d.
	- SVM : The :mod:`sklearn.svm` module includes Support Vector Machine algorithms.
	- SVC : Support Vector Classification. The implementation is based on libsvm.
	- Kernels : The kernel is effectively a similarity measure.
	- decision_function : Gives per-class scores for each sample (or a single score per sample in the binary case).
	- support_vectors_ : 
	- score : Returns the mean accuracy on the given test data and labels.
	- StandardScaler : Standardize features by removing the mean and scaling to unit variance
	- transform : Transform the data based on what is learned from `fit`
	- pca.explained_variance_ 
	- pca.components_
	- fit_transform : Fit the model with X and apply the dimensionality reduction on X.
	- inverse_transform : Transform data back to its original space, i.e., return an input X_original whose transform would be X.
	- KMeans : K-Means clustering
	- fit_predict : Compute cluster centers and predict cluster index for each sample.
	- confusion_matrix : Compute confusion matrix to evaluate the accuracy of a classification
	- accuracy_score : Accuracy classification score.
	- adjusted_rand_score : The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and 
				counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
	- MiniBatchKMeans : Mini-Batch K-Means clustering
	- cross_val_score : Evaluate a score by cross-validation
	- mean_sqaured_error : Mean squared error regression loss
	- pipeline.make_pipeline : Construct a Pipeline from the given estimators.
	- Polynomial Features : Generate a new feature matrix consisting of all polynomial combinations of the features with degree less than
				 or equal to the specified degree.
	- learning_curve.validation_curve : Determine training and test scores for varying parameter values.
	- feature_extraction.CountVectorizer : Convert a collection of text documents to a matrix of token counts
	- TfidfVectorizer : Convert a collection of raw documents to a matrix of TF-IDF features.
	- SGDClassifier : Linear classifiers (SVM, logistic regression, a.o.) with SGD training.
	- RandomizedPCA : Principal component analysis (PCA) using randomized SVD
	- RandomForestRegressor : A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples 
				of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
