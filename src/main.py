# import numpy as np
from model_new import initialize
from ops import normalize,sparsify
from inputs import argparser

import time
import os

if __name__ == '__main__':
    m_opts = argparser()
    print 'Model Options:'
    print m_opts

    m_vars = initialize(m_opts)

