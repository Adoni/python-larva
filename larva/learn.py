import numpy


def learn(source, target, *priors):
    '''
    Learns from some data and adjusts the priors accordingly.

    Combines and updates any supplied priors with learned information.
    '''
    source = numpy.asarray(source)
    target = numpy.asarray(target)

    return source


def apply(source, *priors):
    '''
    Applies learned priors to a source.

    Returns a target based on the source and priors.
    '''
    source = numpy.asarray(source)

    return source
