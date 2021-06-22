#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

import numpy as np
import numpy.linalg as LA
import scipy.optimize as opt
from scipy.spatial import  cKDTree


def optimizeMesh(df, points, atomlist):  # TODO
    Kdr = cKDTree(atomlist)
    for i in range(points.shape[0]):
        pi = points[i]
        ss,neig=Kdr.query(pi,5)
        scenter=atomlist[neig].mean(axis=0)
        dli=pi-scenter

        def funx(x):
            al = pi-x*dli
            return df(np.array([al]))[0]**2

        xs = opt.minimize_scalar(funx,bounds=(0.0, 1.0), method='bounded')

        points[i] -= dli*xs.x

    dpi = df(points)
    center=points.mean(axis=0)
    radil = points-center
    disradil = LA.norm(radil, axis=1)
    EPS = dpi.max()/disradil.mean()
    return EPS
