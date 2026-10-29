# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

'Import-safe quadratic example; register explicitly with an MCP server.'
import math
def quadratic_solver(a, b, c):
    'Return finite real roots of a nondegenerate quadratic equation.'
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (a, b, c)):
        raise ValueError('Quadratic coefficients must be finite numbers')
    if a == 0:
        raise ValueError('Quadratic coefficient must be nonzero')
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return 'Complex roots are not supported by this example'
    root = math.sqrt(discriminant)
    return [(-b+root)/(2*a), (-b-root)/(2*a)]

# Author: Weiliang Zhu; Other main author: Yaming Lv
# Email: wlzchina05@gmail.com
