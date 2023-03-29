import sys 
from deepinv.optim.optim_iterators import *
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)