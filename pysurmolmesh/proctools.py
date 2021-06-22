#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

import numpy as np

import pysurmolmesh.spheremesh as spm
import numpy.linalg as LA
import pysurmolmesh.optimizeMesh as optx
from scipy.spatial import Delaunay, cKDTree


def getSpGen(atomlist, radilist, rho):
    """
    :param atomlist: numpy array shape (natom,3)
    :param rho: radius resolution used in the generation
    :return: xyzpoints, bounds, trig.simplices, eps
    """
    assert (type(atomlist) == type(spm.arant)), " atomlist must be a numpy array"
    assert (atomlist.shape[1] == 3), "dimension of atoms must be 3"
    center = atomlist.mean(axis=0)
    dista = atomlist - center
    radius = LA.norm(dista, axis=1).max()
    radius += radilist.max()
    return spm.genSphereMesh(radius, rho, center)


class VesicleGenerator:
    def __init__(self, atomlist, radiumlist):
        self.atomlist = atomlist
        self.radiumlist = radiumlist
        self.maxradi = radiumlist.max()
        self.Kdr = cKDTree(self.atomlist)

    def df(self, p):  # Critic point TODO
        # direct distance
        rq = self.Kdr.query(p)
        dist = rq[0] - self.radiumlist[rq[1]]  # len(p)*log2(len(atomlist))
        return dist

    def generator(self, rho):
        xyzpoints, bounds, faces = getSpGen(
            self.atomlist, self.radiumlist, rho)

        eps = optx.optimizeMesh(self.df, xyzpoints,self.atomlist)

        trig = Delaunay(xyzpoints)

        bounds = []
        faces = []
        for t in trig.simplices:
            t1 = t[0]
            t2 = t[1]
            t3 = t[2]
            t4 = t[3]
            face1 = [t1, t2, t3]
            face2 = [t1, t2, t4]
            face3 = [t2, t3, t4]
            face4 = [t1, t3, t4]
            d1 = self.df(np.array([xyzpoints[face1].mean(axis=0)]))[0]
            dd = d1
            xface = face1
            d2 = self.df(np.array([xyzpoints[face2].mean(axis=0)]))[0]
            if (d2 > dd):
                dd = d2
                xface = face2
            d3 = self.df(np.array([xyzpoints[face3].mean(axis=0)]))[0]
            if (d3 > dd):
                dd = d3
                xface = face3
            d4 = self.df(np.array([xyzpoints[face4].mean(axis=0)]))[0]
            if (d4 > dd):
                dd = d4
                xface = face4
            st1 = xface[0]
            st2 = xface[1]
            st3 = xface[2]
            faces.append(xface)
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
        internals = np.array(tuple(faces))

        return xyzpoints, bounds, internals, eps
