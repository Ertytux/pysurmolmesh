#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

import numpy as np
import numpy.linalg as LA


def optimizeMesh(df, points, bounds, maxcount=1000):  # TODO
    count = 0
    a1 = np.array([1, 0, 0])
    a2 = np.array([0, 1, 0])
    a3 = np.array([0, 0, 1])
    Forces = points.copy()
    while True:
        vecbound = points[bounds[:, 1]] - points[bounds[:, 0]]
        disbound = LA.norm(vecbound, axis=1)
        bmin = disbound.min()
        bmax = disbound.max()
        h0 = 0.5 * (bmax + bmin)
        EPS = (bmax - bmin) / h0
        delta = h0 * 1.e-3
        dp = df(points)
        for i in range(points.shape[0]):
            fx = (-(df(np.array([points[i]]) + a1 * delta) - dp[i]) / delta)[0]
            fy = (-(df(np.array([points[i]]) + a2 * delta) - dp[i]) / delta)[0]
            fz = (-(df(np.array([points[i]]) + a3 * delta) - dp[i]) / delta)[0]
            ff = fx * fx + fy * fy + fz * fz
            Forces[i, 0] = dp[i] * fx / ff / h0
            Forces[i, 1] = dp[i] * fy / ff / h0
            Forces[i, 2] = dp[i] * fz / ff / h0
        for i in range(bounds.shape[0]):
            st = bounds[i, 0]
            en = bounds[i, 1]
            Fb = (1.0 - disbound[i] / h0) * vecbound[i] / disbound[i]
            Forces[en] += Fb
            Forces[st] -= Fb
        points += Forces * delta
        count += 1
        if EPS < 1.e-2 or count > maxcount:
            break

    return EPS

