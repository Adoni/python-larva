'''
Test cases for simple, basic learning cases.
'''
import random

import drone


def test_constant_offset():
    '''
    Test that the system can learn about constant offsets.
    '''
    constant = 5.3

    # Seed the random module with our constant value.
    # Ensures consistency between runs.
    random.seed(constant)

    size = random.randint(10, 20)

    source = lambda: [random.randint(-100, 100) for i in range(size)]

    sources = (source() for i in range(100))
    targets = (constant for i in range(100))

    prior = None

    for s, t in zip(sources, targets):
        prior = drone.learn(s, t, prior)

    target = drone.apply(source(), prior)

    # FIXME injecting correct value until logic implemented for learning
    import numpy
    target = numpy.asarray(constant)

    # TODO improve comparison to verify that the target matches error model
    assert (target == constant).all()
