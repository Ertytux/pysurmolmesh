#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

import vpython as vp
import numpy as np
from molmeshtool.numvpvector import NpAtoVv

ar = np.array([1.0])


def genAtomView(listpost, listradius, listcolor=None, listbounds=None, boundcolor=vp.color.red):
    """
    Generate a Vpython view of a set of particles
    :param listpost: a numpy array with particles position np.array([x,y,z])
    :param listradius: a numpy array with particles radius
    :param listcolor: optional list of particles colors
    :return: vpython object of list of spheres view
    """
    # validations
    assert (type(listpost) == type(ar)), "first argument is not a numpy array"
    assert (type(listradius) == type(ar)
            ), "second argument is not a numpy array"
    assert (
        listpost.shape[0] == listradius.size), "number of position and radious differ "
    # If all is ok then make spheres view
    n = listpost.shape[0]
    listsphere = []
    defacolor = vp.color.white
    for i in range(n):
        if listcolor is not None:
            defacolor = listcolor[i]
        sph = vp.sphere(pos=NpAtoVv(
            listpost[i]), radius=listradius[i], color=defacolor)
        listsphere.append(sph)
    bx = []
    if listbounds is not None:
        for bt in listbounds:
            bx.append(vp.curve(NpAtoVv(listpost[bt[0]]), NpAtoVv(
                listpost[bt[1]]), color=boundcolor))

    return listsphere, bx


def genAtomViewMPL(pts):
    """
    Show using 3D matplotlib view of particles position
    :param pts: a numpy array with particles position np.array([x,y,z])
    :return: the image reference
    """
    assert (type(pts) == type(ar)), "first argument is not a numpy array"
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='o')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.show()
    return ax
