# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:58:47 2017

@author: Yurez23
"""
import numpy as np
import numpy.linalg as lin
# import matplotlib as mpl
import matplotlib.pyplot as plt

# from matplotlib import mlab, cm

eps = 1e-3
startPoint1 = np.array([-2 * np.sqrt(5), 1])
# print('!!!',startPoint1,'!!!')
startPoint2 = np.array([-1.0, -2.0])
startPoint3 = np.array([-8, 2])
countr = 0
itercountr = 0


def MyFunc1(X):  # Входная функция (квадратичная)
    global countr
    countr += 1
    return (6 * X[0] ** 2 + 3 * X[1] ** 2 - 4 * X[0] * X[1] + 4 * np.sqrt(5) * (X[0] + 2 * X[1]) + 22)


def MyFunc2(X, alpha=100):  # Входная функция (Розенброка)
    global countr
    countr += 1
    return (alpha * (X[0] ** 2 - X[1]) ** 2 + (X[0] - 1) ** 2)


def MyFunc3(X):  # Входная функция (Розенброка)
    global countr
    countr += 1
    return ((X[0] ** 2 + X[1] - 11) ** 2 + (X[0] + X[1] ** 2 - 7) ** 2)


# print('!!!',startPoint1,'!!!')
# def MyFunc2(*args):
#    return(6*args[0]**2 + 3*args[1]**2 - 4*args[0]*args[1] + 4*np.sqrt(5)*(args[0] + 2*args[1]) + 22)
#
# def MyDiff(f,startPoint1):
#    df=(f(startPoint1+eps)-f(startPoint1-eps))/(2*eps)
#    return(df)

def MyDiffx(f, X0):  # Производная по x в точке X0
    eps2 = eps/1000
    xplusdx = np.array([X0[0] + eps2, X0[1]])
    xminusdx = np.array([X0[0] - eps2, X0[1]])
    return ((f(xplusdx) - f(xminusdx)) / (2 * eps2))


def MyDiffy(f, X0):  # Производная по y в точке X0
    eps2 = eps/1000
    yplusdy = np.array([X0[0], X0[1] + eps2])
    yminusdy = np.array([X0[0], X0[1] - eps2])
    return ((f(yplusdy) - f(yminusdy)) / (2 * eps2))


def MyGrad(f, X0):  # Градиент в точке X0
    return (np.array([MyDiffx(f, X0), MyDiffy(f, X0)]))


# def MyNorm(vect):
#    l = len(vect)
#    sum = 0
#    it = 0
#    while it<l:
#        sum += vect[it]**2
#        it += 1
#    return(np.sqrt(sum))

def MyArgMin(f, X0, a, b):
    eps2 = eps/1000
    while np.abs(b - a) > eps2:
        x = (a + b) / 2
        f1 = f(x - eps2, X0)
        f2 = f(x + eps2, X0)
        #        print(f(x,X0))
        if f1 >= f2:
            a = x
        else:
            b = x
            #        print(a,b)
    return (x)


def MyArgMin(f, X0, a, b):
    eps2 = eps/1000
    while np.abs(b - a) > eps2:
        x = (a + b) / 2
        f1 = f(x - eps2)
        f2 = f(x + eps2)
        #        print(f(x,X0))
        if f1 >= f2:
            a = x
        else:
            b = x
            #        print(a,b)
    return (x)


def MyHesse(f, X0):
    H = np.array([[0.0,0.0],[0.0,0.0]])
    eps1=eps/1000
    eps2=eps1**2
    f00=f(X0)
    f10=f(X0 + np.array([eps1, 0]))
    f01=f(X0 + np.array([0, eps1]))
    f11=f(X0 + np.array([eps1, eps1]))
#    H[0,0] = (f00 + f(X0 + 2 * np.array([eps1, 0])) - 2 * f10) / (eps2)
    H[1,0] = H[0,1] = (f00 + f11 - f10 - f01) / (eps2)
#    H[1,1] = (f00 + f(X0 + 2 * np.array([0, eps1])) - 2 * f01) / (eps2)
    H[0,0] = (f(X0 - np.array([eps1, 0])) + f10 - 2 * f00) / (eps2)
    H[1,1] = (f(X0 - np.array([0, eps1])) + f01 - 2 * f00) / (eps2)
    return(H)
#    return (
#        np.array([[(f(X0) + f(X0 + 2 * np.array([eps, 0])) - 2 * f(X0 + np.array([eps, 0]))) / (eps ** 2),
#                   (f(X0) + f(X0 + np.array([eps, eps])) - f(X0 + np.array([eps, 0])) - f(X0 + np.array([0, eps]))) / (
#                       eps ** 2)],
#                  [(f(X0) + f(X0 + np.array([eps, eps])) - f(X0 + np.array([eps, 0])) - f(X0 + np.array([0, eps]))) / (
#                      eps ** 2),
#                   (f(X0) + f(X0 + 2 * np.array([0, eps])) - 2 * f(X0 + np.array([0, eps]))) / (eps ** 2)]]))


def MNS(f, X0):
    def MyFunc1_1(kappa, X0):
        arg = X0 + kappa * antiGrad / normGrad
        return (f(arg))

    X = X0
    XX = np.array([X])
    normGrad = lin.norm(MyGrad(f, X0))
    antiGrad = -MyGrad(f, X0)
    ii = 0
    global itercountr
    while normGrad > eps:
        itercountr += 1
        kappa = MyArgMin(MyFunc1_1, X, 0, 100)
        X = X + kappa * antiGrad / normGrad
        XX = np.append(XX, np.array([X]), axis=0)
        normGrad = lin.norm(MyGrad(f, X))
        antiGrad = -MyGrad(f, X)
        if ii == 1000000:
            print('Too many iterations!!!')
            break
        else:
            ii += 1
            #        print( 'X =', X, 'f(X) =', f(X) )
            #        print( 'antiGrad =', antiGrad, 'normGrad =', normGrad )
    return (X, f(X), XX)


def MDSH(f, X0, kappa0=1, gamma=0.5):
    def MyFunc1_1(kappa, X0):
        arg = X0 + kappa * antiGrad / normGrad
        return (f(arg))

    X = X0
    X1 = X
    XX = np.array([X])
    kappa = kappa0
    normGrad = lin.norm(MyGrad(f, X0))
    antiGrad = -MyGrad(f, X0)
    ii = 0
    global itercountr
    while normGrad > eps:
        while f(X) <= f(X1):
            itercountr += 1
            X1 = X
            X = X + kappa * antiGrad / normGrad
            XX = np.append(XX, np.array([X]), axis=0)
            normGrad = lin.norm(MyGrad(f, X))
            antiGrad = -MyGrad(f, X)
            #            print( 'X =', X, 'f(X) =', f(X) )
            #            print( 'antiGrad =', antiGrad, 'normGrad =', normGrad )
            #            print( 'kappa =', kappa)
            if ii == 1000000:
                print('Too many iterations!!!')
                break
            else:
                ii += 1
        kappa = kappa * gamma
        X1 = X
        X = X + kappa * antiGrad / normGrad
        XX = np.append(XX, np.array([X]), axis=0)
        normGrad = lin.norm(MyGrad(f, X))
        antiGrad = -MyGrad(f, X)
        #        print( 'X =', X, 'f(X) =', f(X) )
        #        print( 'antiGrad =', antiGrad, 'normGrad =', normGrad )
        #        print( 'kappa =', kappa)
        if ii == 1000:
            print('Too many iterations!!!')
            break
        else:
            ii += 1
    return (X, f(X), XX)


def MFR(f, X0):
    def MyFunc1_1(kappa):
        arg = X + kappa * p
        return (f(arg))

    antiGrad1 = -MyGrad(f, X0)
    normGrad1 = lin.norm(antiGrad1)
    p = antiGrad1
    X = X0
    XX = np.array([X])
    kappa = MyArgMin(MyFunc1_1, X, 0, 100)
    # print('kappa =',kappa)
    X = X + kappa * p
    XX = np.append(XX, np.array([X]), axis=0)
    global itercountr
    itercountr += 1

    while normGrad1 > eps:
        antiGrad2 = -MyGrad(f, X)
        normGrad2 = lin.norm(antiGrad2)
        # print('normGrad =',normGrad2)
        if normGrad2 < eps:
            break
        gamma = ((normGrad2 ** 2) / (normGrad1 ** 2)) * (itercountr % 2)
        # print('gamma =', gamma)
        p = gamma * p + antiGrad2
        # print('p =',p)
        kappa = MyArgMin(MyFunc1_1, X, 0, 10)
        # print('kappa =', kappa)
        X = X + kappa * p
        XX = np.append(XX, np.array([X]), axis=0)
        antiGrad1 = antiGrad2
        normGrad1 = normGrad2
        itercountr += 1
        if itercountr >= 1000:
            print('Too many iterations!!!')
            break

    return (X, f(X), XX)


def MPR(f, X0):
    def MyFunc1_1(kappa):
        arg = X + kappa * p
        return (f(arg))

    antiGrad1 = -MyGrad(f, X0)
    normGrad1 = lin.norm(antiGrad1)
    p = antiGrad1
    X = X0
    XX = np.array([X])
    kappa = MyArgMin(MyFunc1_1, X, 0, 100)
    # print('kappa =',kappa)
    X = X + kappa * p
    XX = np.append(XX, np.array([X]), axis=0)
    global itercountr
    itercountr += 1

    while normGrad1 > eps:
        antiGrad2 = -MyGrad(f, X)
        normGrad2 = lin.norm(antiGrad2)
        # print('normGrad =',normGrad2)
        if normGrad2 < eps:
            break
        gamma = (np.dot(antiGrad2 - antiGrad1, antiGrad2) / (normGrad1 ** 2)) * (itercountr % 2)
        # print('gamma =', gamma)
        p = gamma * p + antiGrad2
        # print('p =',p)
        kappa = MyArgMin(MyFunc1_1, X, 0, 10)
        # print('kappa =', kappa)
        X = X + kappa * p
        XX = np.append(XX, np.array([X]), axis=0)
        antiGrad1 = antiGrad2
        normGrad1 = normGrad2
        itercountr += 1
        if itercountr >= 1000:
            print('Too many iterations!!!')
            break

    return (X, f(X), XX)


def MSG(f, X0):
    def MyFunc1_1(kappa):
        arg = X + kappa * p
        return (f(arg))

    antiGrad1 = -MyGrad(f, X0)
    normGrad1 = lin.norm(antiGrad1)
    p = antiGrad1
    X = X0
    XX = np.array([X])
    kappa = MyArgMin(MyFunc1_1, X, 0, 100)
    # print('kappa =',kappa)
    X = X + kappa * p
    XX = np.append(XX, np.array([X]), axis=0)
    global itercountr
    itercountr += 1

    while normGrad1 > eps:
        antiGrad2 = -MyGrad(f, X)
        normGrad2 = lin.norm(antiGrad2)
        # print('normGrad =',normGrad2)
        if normGrad2 < eps:
            break
        HX=MyHesse(f,X)
        HXp=np.dot(HX,p)
        gamma = -(np.dot(HXp,antiGrad2)/np.dot(HXp,p)) * (itercountr % 2)
        # print('gamma =', gamma)
        p = gamma * p + antiGrad2
        # print('p =',p)
        kappa = MyArgMin(MyFunc1_1, X, 0, 10)
        # print('kappa =', kappa)
        X = X + kappa * p
        XX = np.append(XX, np.array([X]), axis=0)
        antiGrad1 = antiGrad2
        normGrad1 = normGrad2
        itercountr += 1
        if itercountr >= 1000:
            print('Too many iterations!!!')
            break

    return (X, f(X), XX)

def MN(f, X0):
    X = X0
    XX = np.array([X])
    global itercountr
#    H=np.array([[12,-4],[-4,6]])
    w = -MyGrad(f, X)
    nw = lin.norm(w)
    while nw > eps:
        H = MyHesse(f, X)
        p = lin.solve(H, w)
        X = X + p
        XX = np.append(XX, np.array([X]), axis=0)
        itercountr += 1
        w = -MyGrad(f, X)
        nw = lin.norm(w)
    return(X, f(X), XX)

def MNwED(f, X0):
    def myFunc(kappa):
        return(f(X + kappa * p))
    X = X0
    XX = np.array([X])
    w = -MyGrad(f, X)
    nw = lin.norm(w)
    itercounter = 0
    while nw > eps:
        H = MyHesse(f, X)
        p = lin.solve(H, w)
        kappa = MyArgMin(myFunc, X0, 0, 10)
        X = X + kappa * p
        XX = np.append(XX, np.array([X]), axis=0)
        itercounter += 1
        w = -MyGrad(f, X)
        nw = lin.norm(w)
        # print('p =', p)
        # print('X =', X)
        # print('kappa =', kappa)
        print('||w|| =', nw)
        if itercounter >= 1000:
            print('To many iterations!!!')
            break
    print('itercounter =', itercounter)
    return(X, f(X), XX)

def MyBounds(points):
    xleft = np.min(points[:, 0])
    xright = np.max(points[:, 0])
    yleft = np.min(points[:, 1])
    yright = np.max(points[:, 1])
    #    print('xleft =',xleft)
    #    print('xright =',xright)
    #    print('yleft =',yleft)
    #    print('yright =',yright)
    #    print()
    xmid = (xleft + xright) / 2
    ymid = (yleft + yright) / 2
    #    print('xmid =',xmid)
    #    print('ymid =',ymid)
    #    print()
    xdiap = xright - xleft
    ydiap = yright - yleft
    #    print('xdiap =',xdiap)
    #    print('ydiap =',ydiap)
    #    print()
    maxdiap = np.max([xdiap, ydiap])
    #    print('maxdiap =',maxdiap)
    #    print()
    xleft = xmid - 1.2 * maxdiap / 2
    xright = xmid + 1.2 * maxdiap / 2
    yleft = ymid - 1.2 * maxdiap / 2
    yright = ymid + 1.2 * maxdiap / 2
    #    print('xleft =',xleft)
    #    print('xright =',xright)
    #    print('yleft =',yleft)
    #    print('yright =',yright)
    #    print()
    xBound = [xleft, xright]
    yBound = [yleft, yright]
    return (xBound, yBound)


def MyContourPlot(f, xMinMax, yMinMax):
    delta = (xMinMax[1] - xMinMax[0]) / 20
    x = np.arange(xMinMax[0], xMinMax[1], delta)
    y = np.arange(yMinMax[0], yMinMax[1], delta)
    X, Y = np.meshgrid(x, y)
    Z = f([X, Y])
    #    plt.figure()
    CS = plt.contourf(X, Y, Z)
    plt.colorbar(CS)


#    plt.grid()
#    plt.clabel(CS, inline=1, fontsize=10)
#    plt.title('Simplest default with labels')

def MyContourScatterArrowPlot(func, points):
    xBound, yBound = MyBounds(points)
    # xBound = np.array([-10,10])
    # yBound = np.array([-10,10])
    arrow_head_width = (xBound[1] - xBound[0]) / 50
    arrow_line_width = (xBound[1] - xBound[0]) / 1000
    h = 7
    fig1 = plt.figure(figsize=(1.2 * h, h))
    MyContourPlot(func, xBound, yBound)
    ppoint = points[0]
    plt.scatter(points[0, 0], points[0, 1], color='black')
    inlen = lin.norm(points[1] - points[0])
    for point in points[1:, :]:
        curlen = lin.norm(point - ppoint)
        #        if 100*curlen < inlen:
        #            break
        plt.scatter(point[0], point[1], color='black')
        plt.arrow(ppoint[0], ppoint[1], point[0] - ppoint[0], point[1] - ppoint[1], width=arrow_line_width, head_width=arrow_head_width,
                  color='red', length_includes_head=True)
        ppoint = point
    plt.scatter(points[-1, 0], points[-1, 1], color='black')
    plt.show()

# Xmin1, fmin1, Xit1 = MNwED(MyFunc1, startPoint1)
# print('Xmin =', np.around(Xmin1, decimals=3), 'fmin =', np.around(fmin1, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc1, Xit1[:, :])

Xmin2, fmin2, Xit2 = MNwED(MyFunc2, startPoint2)
print('Xmin =', np.around(Xmin2, decimals=3), 'fmin =', np.around(fmin2, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
MyContourScatterArrowPlot(MyFunc2, Xit2[:, :])

# Xmin11, fmin11, Xit11 = MNS( MyFunc1, startPoint1 )
# print('Xmin' , np.around(Xmin11, decimals=3),'fmin =', np.around(fmin11, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc1, Xit11[:,:])

# Xmin12, fmin12, Xit12 = MDSH( MyFunc1, startPoint1 )
# print('Xmin' , np.around(Xmin12, decimals=3),'fmin =', np.around(fmin12, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc1, Xit12[:,:])

# Xmin21, fmin21, Xit21 = MNS( MyFunc2, startPoint2 )
# print('Xmin' , np.around(Xmin21, decimals=3),'fmin =', np.around(fmin21, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc2, Xit21[:,:])

# Xmin22, fmin22, Xit22 = MDSH( MyFunc2, startPoint2 )
# print('Xmin' , np.around(Xmin22, decimals=3),'fmin =', np.around(fmin22, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc2, Xit22[:,:])

# Xmin31, fmin31, Xit31 = MNS( MyFunc3, startPoint3 )
# print('Xmin' , np.around(Xmin31, decimals=3),'fmin =', np.around(fmin31, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc3, Xit31[:,:])

# Xmin32, fmin32, Xit32 = MDSH( MyFunc3, startPoint3 )
# print('Xmin' , np.around(Xmin32, decimals=3),'fmin =', np.around(fmin32, decimals=3))
# print('funccounter =', countr, 'itercounter =', itercountr)
# MyContourScatterArrowPlot(MyFunc3, Xit32[:,:])


# def testMyFunc1(X):
#    return(X[0]**2+X[1]**2)
#
# testx=np.array([ [ [0,0.5,1], [0,0.5,1], [0,0.5,1] ], [ [0,0,0], [0.5,0.5,0.5], [1,1,1] ] ])
# testf=testMyFunc1(testx)
# print(testx)
# print('================================')
# print(testf)

# agrad1 = -MyGrad(MyFunc1, startPoint1)
# argmin1 = MyArgMin(MyFunc1_1, startPoint1, 0, 100)

# (Xmin, fmin) = MDSH( MyFunc1, startPoint1 , 1, 0.5)
# print('fmin =', fmin, 'countr =', countr1)
