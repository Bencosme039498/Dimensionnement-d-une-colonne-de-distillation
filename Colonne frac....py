# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:57:21 2021

@authors: Erny ENCARNACION and Juan BENCOSME.

"""
import numpy as np
from scipy.optimize import fsolve

'COMPOSICION ALIMENTATION Z'
W = 66.05 # flux dans le residu
D_dest = 33.95 # flux dans le destillat
z = [0.003, 0.004, 0.333, 0.23, 0.29, 0.14]

c1 = [4.19, -832, 7.17e-4] # methane
c2 = [4.11,-1210, 1.32e-3] # ethylene
c3 = [4.09, -1270, 1.11e-3 ] # ethane
c4 = [4.99, -1680, -4.47e-4] # propane
c5 = [9.63, -2950, -5.71e-3] # buthane
c6 = [6.64, -3110, -5.69e-4] # hexane
' PUNTO B- INCISO A '
npoints =85
T = np.linspace(288,473,npoints)
Cdestilat=[0.009,0.012,0.932,0.047]   # Composition Destilat
Cresidu=[0.02513, 0.32384, 0.43906, 0.21196] # Composition Residu
K = np.zeros(6)
Td = np.zeros(npoints)
Tb = np.zeros(npoints)
Error=1e-2
for i in range(npoints):
    K[0] = np.exp(c1[0] + c1[1]/T[i] + c1[2]*T[i]) # methane
    K[1] = np.exp(c2[0] + c2[1]/T[i] + c2[2]*T[i]) # ethylene
    K[2] = np.exp(c3[0] + c3[1]/T[i] + c3[2]*T[i]) # ethane
    K[3] = np.exp(c4[0] + c4[1]/T[i] + c4[2]*T[i]) # propane
    
    
    Td[i]=Cdestilat[0]/K[0]+ Cdestilat[1]/K[1] + Cdestilat[2]/K[2] + Cdestilat[3]/K[3]
    
    if abs(Td[i]-1)<Error:
        Tb1=T[i]
        print('Temperature du rosee du distillat[K] =',T[i])
        print('Temperature du rosee du distillat[C] =',T[i]-273.15)
        break
    
      
for i in range (npoints):  
     K[2] = np.exp(c3[0] + c3[1]/T[i] + c3[2]*T[i]) # ethane
     K[3] = np.exp(c4[0] + c4[1]/T[i] + c4[2]*T[i]) # propane
     K[4] = np.exp(c5[0] + c5[1]/T[i] + c5[2]*T[i]) # buthane
     K[5] = np.exp(c6[0] + c6[1]/T[i] + c6[2]*T[i]) # hexane
     Tb[i]=Cresidu[0]*K[2]+ Cresidu[1]*K[3] + Cresidu[2]*K[4] + Cresidu[3]*K[5]
    
     if abs(Tb[i]-1)<Error:
         
         print('\nTemperature du bulle du residu[K] =',T[i])
         print('Temperature du bulle du residu[C] =',T[i]-273.15)
         Tb2=T[i]
         break
K1=np.zeros(6) 
alfad=np.zeros(6)       
K1[0] = np.exp(c1[0] + c1[1]/Tb1 + c1[2]*Tb1) # methane
K1[1] = np.exp(c2[0] + c2[1]/Tb1 + c2[2]*Tb1) # ethylene
K1[2] = np.exp(c3[0] + c3[1]/Tb1 + c3[2]*Tb1) # ethane
K1[3] = np.exp(c4[0] + c4[1]/Tb1 + c4[2]*Tb1) # propane
K1[4] = np.exp(c5[0] + c5[1]/Tb1 + c5[2]*Tb1) # buthane
K1[5] = np.exp(c6[0] + c6[1]/Tb1 + c6[2]*Tb1) # hexane
Kclr =K1[3]                                   # Cle Lourd par rapport T Distillat     
alfad[0]=K1[0]/Kclr
alfad[1]=K1[1]/Kclr
alfad[2]=K1[2]/Kclr
alfad[3]=K1[3]/Kclr
alfad[4]=K1[4]/Kclr
alfad[5]=K1[5]/Kclr
print('\nVolatilites Relatives Destillat [Methane]=',alfad[0])
print('Volatilites Relatives Destillat  [Ethylene]=',alfad[1])
print('Volatilites Relatives Destillat [Ethane]=',alfad[2])
print('Volatilites Relatives Destillat [Propane]=',alfad[3])
print('Volatilites Relatives Destillat  [Buthane]=',alfad[4])
print('Volatilites Relatives Destillat  [Hexane]=',alfad[5])


K2=np.zeros(6) 
alfab=np.zeros(6) 
K2[0] = np.exp(c1[0] + c1[1]/Tb2 + c1[2]*Tb2) # methane
K2[1] = np.exp(c2[0] + c2[1]/Tb2 + c2[2]*Tb2) # ethylene
K2[2] = np.exp(c3[0] + c3[1]/Tb2 + c3[2]*Tb2) # ethane
K2[3] = np.exp(c4[0] + c4[1]/Tb2 + c4[2]*Tb2) # Propane
K2[4] = np.exp(c5[0] + c5[1]/Tb2 + c5[2]*Tb2) # buthane
K2[5] = np.exp(c6[0] + c6[1]/Tb2 + c6[2]*Tb2) # hexane
Kclb=np.exp(c4[0] + c4[1]/Tb2 + c4[2]*Tb2)   # Cle Lourd par rapport T Residu 
alfab[0]=K2[0]/Kclb
alfab[1]=K2[1]/Kclb
alfab[2]=K2[2]/Kclb
alfab[3]=K2[3]/Kclb
alfab[4]=K2[4]/Kclb
alfab[5]=K2[5]/Kclb

print('\nVolatilites Relatives Residu  [Methane]=',alfab[0])
print('Volatilites Relatives Residu  [Ethylene]=',alfab[1])
print('Volatilites Relatives Residu  [Ethane]=',alfab[2])
print('Volatilites Relatives Residu  [Propane]=',alfab[3])
print('Volatilites Relatives Residu  [Buthane]=',alfab[4])
print('Volatilites Relatives Residu  [Hexane]=',alfab[5])

alfamoy=np.zeros(6)
alfamoy[0]=(alfab[0]*alfad[0])**(1/2)
alfamoy[1]=(alfab[1]*alfad[1])**(1/2)
alfamoy[2]=(alfab[2]*alfad[2])**(1/2)
alfamoy[3]=(alfab[3]*alfad[3])**(1/2)
alfamoy[4]=(alfab[4]*alfad[4])**(1/2)
alfamoy[5]=(alfab[5]*alfad[5])**(1/2)
print('\nVolatilites Relatives Moyenne [Methane]=',alfamoy[0])
print('Volatilites Relatives Moyenne [Ethylene]=',alfamoy[1])
print('Volatilites Relatives Moyenne [Ethane]=',alfamoy[2])
print('Volatilites Relatives Moyenne [Propane]=',alfamoy[3])
print('Volatilites Relatives Moyenne [Buthane]=',alfamoy[4])
print('Volatilites Relatives Moyenne [Hexane]=',alfamoy[5])

'TEST DE SHIRAS POUR LE CR'

TRD_CV = 31.64/33.3 # Taux recuperation ethane dans le destillat
TRD_CL = 1.61/23 # TUX recuperation propane dans le destillat
alpha_CV = alfamoy[2]
TRD= np.zeros(6)
composant = ['[Methane]','[Ethylene]','[Ethane]', '[Propane]', ' [Buthane]', '[Hexane]' ]

for i in range(6):
    TRD[i] = ((alfamoy[i]-1)*TRD_CV + (alpha_CV - alfamoy[i])*TRD_CL)/(alpha_CV -1)
    
    if TRD[i] > 1.01 or TRD[i]<-0.01 :
        print('\nLe constituant',composant[i],'TRD =',TRD[i], "n'est pas reparti")
    elif 0.01<TRD[i]<0.99 :
        print('\nLe constituant',composant[i],'TRD =',TRD[i], "est certaiment reparti" )
    elif -0.01<TRD[i]<0.01 or 0.99<TRD[i]<1.01 :
        print('\nLe constituant',composant[i],'TRD =',TRD[i], " n'est pas reparti" )
        
'CONDITION THERMODYNAMIQUE DE ALIMENTATION'

Ta = 360 # [K] temperature alimentation
K3=np.zeros(6)
K3[0] = np.exp(c1[0] + c1[1]/Ta + c1[2]*Ta) # methane
K3[1] = np.exp(c2[0] + c2[1]/Ta + c2[2]*Ta) # ethylene
K3[2] = np.exp(c3[0] + c3[1]/Ta + c3[2]*Ta) # ethane
K3[3] = np.exp(c4[0] + c4[1]/Ta + c4[2]*Ta) # propane
K3[4] = np.exp(c5[0] + c5[1]/Ta + c5[2]*Ta) # buthane
K3[5] = np.exp(c6[0] + c6[1]/Ta + c6[2]*Ta) # hexane

Kizi = np.zeros(6) # ki*zi
Ki_zi = np.zeros(6) # ki/zi
for i in range(6):
    Kizi[i]=K3[i]*z[i]
    Ki_zi[i] = z[i]/K3[i]

if np.sum(Kizi)< 1 and np.sum(Ki_zi) >1 :
    print('\nLe melange multicomposant est LIQUIDE-SOUS REFROIDI')
    print('\nKiZi =', np.sum(Kizi), "\nZi/Ki = ",np.sum(Ki_zi)  )
elif np.sum(Kizi) == 1 and np.sum(Ki_zi) >1 :
    print('\nLe melange multicomposant est LIQUIDE-SATURE')
    print('\nKiZi =', np.sum(Kizi), "\nZi/Ki = ",np.sum(Ki_zi)  )
elif np.sum(Kizi) > 1 and np.sum(Ki_zi) >1 :
    print('\nLe melange multicomposant est LIQUIDE-VAPEUR')
    print('\nKiZi =', np.sum(Kizi), "\nZi/Ki = ",np.sum(Ki_zi)  )
elif np.sum(Kizi) > 1 and np.sum(Ki_zi) == 1 :
    print('\nLe melange multicomposant est VAPEUR SATUREE')
    print('\nKiZi =', np.sum(Kizi), "\nZi/Ki = ",np.sum(Ki_zi)  )
elif np.sum(Kizi) > 1 and np.sum(Ki_zi) < 1 :
   
    print('\nLe melange multicomposant est  VAPEUR SURCHAUFFEE')
    print('\nKiZi =', np.sum(Kizi), "\nZi/Ki = ",np.sum(Ki_zi)  )

'TAUX DE AUGMENTATION DU DEBIT VAPEUR'

D = np.zeros(6)
D1 = np.zeros(6)
D2 = np.zeros(6)
def func(v): # fucntion of feed    
    F = np.zeros(1)
    v=np.zeros(2)
    for i in range(6):
        D1[i] = (K3[i]-1)*z[i]
        D2[i] = 1+(K3[i]-1)*v[0]*v[1]
        D[i] = D1[i]/D2[i]
    F[0] = np.sum(D)
    
    return F

guess = np.array([1.2]) # initial guees values

v = fsolve(func,guess) # solver

# print the solution
print('\nFraction vapeur = ', v[0])

'TAUX REFLUX MINIMUN'

D_div = np.zeros(6)
D_1 = np.zeros(6)
D_2 = np.zeros(6)
def func(u): # fucntion    
    R = np.zeros(1)
    for i in range(6):
        D_1[i] = alfamoy[i]*z[i]
        D_2[i] =alfamoy[i]- u[0]
        D_div[i] =  D_1[i]/ D_2[i]
    R[0] = np.sum(D_div)-v[0]
    return R

guess = np.array([1.4]) # initial guees values

u = fsolve(func,guess) # solver


if u[0] < alfamoy[2]:
    R_div = np.zeros(6)
    R_1 = np.zeros(6)
    R_2 = np.zeros(6)
    print('\n' )
    print('uk = ', u[0] )
    print('la condition (1< u < alpha_CV) est satisfaite: ', '\nu = ', u[0], '\nalpha_CV = ', alfamoy[2]  )
    for i in range(4):
        R_1[i] = alfamoy[i]*Cdestilat[i]
        R_2[i] = alfamoy[i]- u[0]
        R_div[i] =  R_1[i]/R_2[i]
    Rmin = sum(R_div) - 1
    print('Taux de reflux minimun Rmin = ', Rmin )
        
"NOMBRE D'ETAPES THEORIQUE MINIMUN"   

TRD_CV = 31.64/33.3 # Taux recuperation ethane dans le destillat
TRD_CL = 1.61/23 # TauX recuperation propane dans le destillat
TRW_CL = 21.39/23 # Taux recuperation propane dans le residu
TRW_CV = 1.66/33.3 # taux recuperation ethane dans le residu
alpha_CV = alfamoy[2]
NET_min =((np.log(alpha_CV))**(-1))*np.log((TRD_CV*TRW_CL)/(TRW_CV*TRD_CL))
print('\nNombre etapes theorique min NETmin = ', NET_min)

" NOMBRE D'ETAPES (NET) " 
print("\nNOMBRE D'ETAPES (NET):")
R_reel = 1.5*Rmin # taux de reflux reel
X =  (R_reel - Rmin)/(R_reel +1) 
Y = 0.75*(1-X**(0.5668))
NET = (-Y -NET_min)/(Y-1)
print('\nNombre etapes theorique (NET) = ', NET)

" PLATE D'ALIMENTATION "    
print("\nPLATE D'ALIMENTATION:")
x_cl = z[3]/((1-v[0])+v[0]*K3[3])  # fraction propane dans le liquide alimentation
x_cv = z[2]/((1-v[0])+v[0]*K3[2]) # fraction ethane dans le liquide alimentation
print('\nx_CL =',x_cl)
print('x_CV', x_cv)
NET_ratio = np.exp(0.206*np.log((W/D_dest)*(x_cl/x_cv)*(Cresidu[0]/Cdestilat[3])**(2)))
print('NETres/NETesp = ', NET_ratio )
N = np.round(NET)-1 # se le resta el plato del reboiler = 1
N_feed = N/(1+NET_ratio) # ver pagina 798 de ray sinnot.
print("Le plateau d'alimentation est : ", np.round(N_feed))
