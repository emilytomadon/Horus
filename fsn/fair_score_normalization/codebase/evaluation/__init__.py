from . import evaluate_models
from . import model_iterators
from . import rfw_iterator

"""
Add your iterators to this dictionary.
"""
ITERATORS = {
            "example" : model_iterators.ExampleIterator,
            "rfw" : rfw_iterator.RFWIterator
            }