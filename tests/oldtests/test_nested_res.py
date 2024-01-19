import dynesty
from dynesty import utils
from dynesty import DynamicNestedSampler

dres = DynamicNestedSampler.restore("test_linear_regression.txt")
