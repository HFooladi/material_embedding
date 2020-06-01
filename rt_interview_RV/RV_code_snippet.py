
import numpy

# may need to adjust the below function if it is not working correctly for your w2v model
def make_fingerprint_matrix(w2v_model, concepts):
    vectors = [w2v_model.wv[c] if c in w2v_model.wv else numpy.zeros(w2v_model.vector_size) for c in concepts]
    fingerprint_matrix = numpy.array([v for v in vectors])
    return fingerprint_matrix


def get_dot_product_of_arrays(x, y):
    return numpy.trace(numpy.dot(x, numpy.transpose(y)))

	
def calculate_rv_coefficient_of_arrays(x, y, normalize=True):
    """
    :param x, y:    numpy.ndarray objects
                    must be two-dimensional arrays with the same number of columns
                    [for the sake of concreteness, let us say that
                        the shape of x is n-by-k,
                        the shape of y is m-by-k]
    :param normalize:   if True calculate denominator according to definition
                        if False it is assumed that x and y are scaled in such a way, that denominator=1
                        (E.g. by constructing x, y using make_fingerprint_matrix with normalize=True)
    :return:        rv(x, y) as defined on https://en.wikipedia.org/wiki/RV_coefficient
    """
    # compute auxiliary quantities
    x_tr = numpy.transpose(x)  # k-by-n
    y_tr = numpy.transpose(y)  # k-by-m
    x_times_y_tr = numpy.dot(x, y_tr)  # n-by-m
    # compute the numerator in the definition of the rv coefficient
    numerator = get_dot_product_of_arrays(x_times_y_tr, x_times_y_tr)
    denominator = 1.
    if normalize:
        x_times_x_tr = numpy.dot(x, x_tr)  # n-by-n
        y_times_y_tr = numpy.dot(y, y_tr)  # m-by-m
        # compute the denominator in the definition of the rv coefficient
        factor_1 = get_dot_product_of_arrays(x_times_x_tr, x_times_x_tr)
        factor_2 = get_dot_product_of_arrays(y_times_y_tr, y_times_y_tr)
        denominator = numpy.sqrt(factor_1 * factor_2)
    if denominator != 0:
        rv_coefficient = numerator / denominator
    else:
        rv_coefficient = 0
    return rv_coefficient


def calculate_rv_coefficient(reference_words_list, reference_word_model_matrix, w2v_model_to_compare):
	# for reference_words_list use the words from ref_vocab.json
    words_matrix_1 = reference_word_model_matrix  # load using pickle python module from file ref_vocab_matrix.pkl
    words_matrix_2 = make_fingerprint_matrix(w2v_model_to_compare, reference_words_list)
    return calculate_rv_coefficient_of_arrays(words_matrix_1, words_matrix_2)