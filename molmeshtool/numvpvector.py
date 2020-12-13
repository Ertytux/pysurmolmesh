#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

from vpython import vector
from numpy import array


def VvtoNpA(inx):
    """
    Convert from vpython.vector to numpy.array
    :param inx:
    :return numpy array([x,y,z]):
    """
    return array(inx.value)


def NpAtoVv(inx):
    """
    Convert from numpy.array to vpython.vector
    :param inx:
    :return vpython.vector(x,y,z):
    """
    return vector(inx[0], inx[1], inx[2])
