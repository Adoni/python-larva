import larva
import numpy

def test_learn_api():
    '''
    Test the learn API adheres to specs.
    '''
    source = [1, 2, 3]
    target = [0, 1]
    priors = []

    larva.learn(source, target, *priors)


def test_apply_api():
    '''
    Test the apply API adheres to specs.
    '''
    source = [1, 2, 3]
    priors = []

    larva.apply(source, *priors)


def test_learn_returns_ndarray():
    '''
    Test that learn returns a numpy ndarray.
    '''
    source = [1, 2, 3]
    target = [0, 1]
    priors = []

    learned = larva.learn(source, target, *priors)

    assert isinstance(learned, numpy.ndarray)


def test_apply_returns_ndarray():
    '''
    Test that apply returns a numpy ndarray.
    '''
    source = [1,2,3]
    priors = []

    applied = larva.apply(source, *priors)

    assert isinstance(applied, numpy.ndarray)
