from .multiobjective.constrained import Srinivas, Tanaka
from .multiobjective.dtlz import DTLZ1, DTLZ2
from .multiobjective.lz09 import LZ09_F2
from .multiobjective.dachuang import Profit # 大创投资组合问题
from .multiobjective.unconstrained import Kursawe, Fonseca, Schaffer, Viennet2
from .multiobjective.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from .multiobjective.mop_m2m import MOP1, MOP2, MOP3, MOP4, MOP5, MOP6, MOP7
from .singleobjective.unconstrained import OneMax, Sphere


__all__ = [
    'Srinivas', 'Tanaka',
    'Kursawe', 'Fonseca', 'Schaffer', 'Viennet2',
    'DTLZ1', 'DTLZ2',
    'ZDT1', 'ZDT2', 'ZDT3', 'ZDT4', 'ZDT6',
    'LZ09_F2',
    'OneMax', 'Sphere'
]
