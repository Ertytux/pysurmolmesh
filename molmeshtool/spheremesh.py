#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

from scipy.spatial import ConvexHull
import numpy as np
from molmeshtool.optimizeMesh import optimizeMesh1, optimizeMesh2

ant = 1.45

arant = np.array([ant, ant, ant])


def genSphereMesh(radius, rho, center=np.array([0.0, 0.0, 0.0])):
    """
    Generate a mesh of spherical surface of bounded point
    :param radius: radius of the sphere
    :param rho: radius resolution used in the generation
    :param center: optional center of sphere as np.array([x,y,z]), default 0,0,0
    :return: np.array  of mesh vertices, list of bonds as list([starts,ends],...)
    """
    assert (type(center) == type(arant)), "Center must be a np.array of (:,3)"
    assert (radius > 0.0), "radius must be  positive "
    assert (type(rho) is int and rho > 1), "\\rho must be integer and >1"

    sita = np.pi / (2 * rho)
    dr = 2.0 * radius * np.sin(sita / 2.0)
    # add stremes
    xyzpoints = [[0, 0, radius], [0, 0, -radius]]
    for sp in range(1, 2 * rho):
        sitap = sp * sita
        sn1 = np.sin(sitap)
        cn1 = np.cos(sitap)
        depsi = 2.0 * np.arcsin(dr / (2 * radius * sn1))
        mi = int(2.0 * np.pi / depsi)
        for ps in range(mi):
            psi = ps * depsi
            sn2 = np.sin(psi)
            cn2 = np.cos(psi)
            x = radius * cn2 * sn1
            y = radius * sn2 * sn1
            z = radius * cn1
            xyzpoints.append([x, y, z])

    xyzpoints = np.array(xyzpoints) + center

    trig = ConvexHull(xyzpoints)
    bounds = []
    # ordered bounds
    for tr in trig.simplices:
        st1 = tr[0]
        st2 = tr[1]
        st3 = tr[2]
        a1 = [min(st1, st2), max(st1, st2)]
        a2 = [min(st2, st3), max(st2, st3)]
        a3 = [min(st1, st3), max(st1, st3)]
        bounds.append(a1)
        bounds.append(a2)
        bounds.append(a3)
    # eliminate duplicate
    bounds = set(tuple(i) for i in bounds)
    # make iterable but  no modificable
    bounds = np.array(tuple(bounds))

    # stabilize shape
    def df(p): return np.sqrt(((p - center) ** 2).sum(1)) - radius
    eps = optimizeMesh1(df, xyzpoints, bounds)
    eps = optimizeMesh2(df, xyzpoints)

    return xyzpoints, bounds, trig.simplices, eps
