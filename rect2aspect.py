import sympy as sm

"""

        D

A
               C
       B

"""

# World Space (Unknown)

wAx = sm.symbols('wAx')
wAy = sm.symbols('wAy')
wAz = sm.symbols('wAz')

wBx = sm.symbols('wBx')
wBy = sm.symbols('wBy')
wBz = sm.symbols('wBz')

wCx = sm.symbols('wCx')
wCy = sm.symbols('wCy')
wCz = sm.symbols('wCz')

wDx = sm.symbols('wDx')
wDy = sm.symbols('wDy')
wDz = sm.symbols('wDz')

# Image Space (Known)

iAx = sm.symbols('iAx')
iAy = sm.symbols('iAy')

iBx = sm.symbols('iBx')
iBy = sm.symbols('iBy')

iCx = sm.symbols('iCx')
iCy = sm.symbols('iCy')

iDx = sm.symbols('iDx')
iDy = sm.symbols('iDy')

f = sm.symbols('f')

# Constraints

cons_wAx = sm.Eq(wAx / wAz, iAx / f)
cons_wAy = sm.Eq(wAy / wAz, iAy / f)

cons_wCx = sm.Eq(wCx / wCz, iCx / f)
cons_wCy = sm.Eq(wCy / wCz, iCy / f)

cons_wDx = sm.Eq(wDx / wDz, iDx / f)
cons_wDy = sm.Eq(wDy / wDz, iDy / f)


cons_wBx = sm.Eq(wBx / wBz, iBx / f)
cons_wBy = sm.Eq(wBy / wBz, iBy / f)
cons_wBz = sm.Eq(wBz, 1)

cons_perp1 = sm.Eq(0, (wAx - wBx) * (wCx - wBx) + (wAy - wBy) * (wCy - wBy) + (wAz - wBz) * (wCz - wBz))
cons_perp2 = sm.Eq(0, (wAx - wDx) * (wCx - wDx) + (wAy - wDy) * (wCy - wDy) + (wAz - wDz) * (wCz - wDz))

cons_rel_wDx = sm.Eq(wAx - wBx + wCx, wDx)
cons_rel_wDy = sm.Eq(wAy - wBy + wCy, wDy)
cons_rel_wDz = sm.Eq(wAz - wBz + wCz, wDz)

a = sm.nonlinsolve(
    [
    cons_perp1, cons_perp2, 
    cons_wBx, cons_wBy, cons_wBz, 
    cons_wAx, cons_wAy, 
    cons_wCx, cons_wCy, 
    cons_wDx, cons_wDy, 
    cons_rel_wDx, cons_rel_wDy, cons_rel_wDz
    ], 
    
    [wAx, wAy, wAz
     wBx, wBy, wBz, 
     wCx, wCy, wCz, 
     wDx, wDy, wDz])

print type(a), a






