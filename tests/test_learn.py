import drone
import numpy

def test_learn_api():
    '''
    Test the learn API adheres to specs.
    '''
    source = [1, 2, 3]
    target = [0, 1]
    priors = []

    drone.learn(source, target, *priors)


def test_apply_api():
    '''
    Test the apply API adheres to specs.
    '''
    source = [1, 2, 3]
    priors = []

    drone.apply(source, *priors)


def test_learn_returns_ndarray():
    '''
    Test that learn returns a numpy ndarray.
    '''
    source = [1, 2, 3]
    target = [0, 1]
    priors = []

    learned = drone.learn(source, target, *priors)

    assert isinstance(learned, numpy.ndarray)


def test_apply_returns_ndarray():
    '''
    Test that apply returns a numpy ndarray.
    '''
    source = [1,2,3]
    priors = []

    applied = drone.apply(source, *priors)

    assert isinstance(applied, numpy.ndarray)
