#  Copyright (c) 2019.
#  @author Edisel Navas Conyedo
#  @email ertydp@gmail.com
#  This piece of code is distributed under GPLv2 (GNU GENERAL PUBLIC LICENSE Version 2)
#  Copyright terms and conditions as in http://www.gnu.org/licenses/gpl-2.0.html
#
#  -*- coding: utf8 -*-

import pybel
import numpy as np
from mendeleev import element

def loadPDB(filename):
    str = pybel.readfile("PDB",filename)
    atomtypes = []
    atompos = []
    atomradi = []
    for molecule in str:
        for atom in molecule.atoms:
            atomtypes.append(atom.type)
            atompos.append([atom.coords[0], atom.coords[1], atom.coords[2]])
            #TODO
            numx=0.95
            atomradi.append(numx)
            #END TODO
    return atomtypes, np.array(atompos), np.array(atomradi)