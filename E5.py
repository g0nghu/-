'''
Qv Weixiang
2022012308
240314
'''

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimSun']#中文

# 计算传播参数波速
def wave_velocity(eps,mu):
    c = 299792458
    v = c / (np.sqrt(eps*mu))
    return v

# 计算传播矩阵
def pro_matrix(alpha,beta,Z0,l):
    gamma = alpha + 1j*beta
    A = np.empty([2,2],dtype=complex)
    A[0,0] = np.cosh(gamma*l)
    A[0,1] = Z0*np.sinh(gamma*l)
    A[1,0] = np.sinh(gamma*l) / Z0
    A[1,1] = A[0,0]
    return A

# 计算输出电压，相位差，传输效率
def cal_phs_eff(AT,Z0,v0):
    vl = v0 / (AT[0,0]+AT[1,1]+AT[0,1]/Z0+AT[1,0]*Z0)
    phs = np.angle(vl)
    eff = (2*np.abs(vl)/v0)**2
    return vl,phs,eff

# 计算群速度
def cal_vg(Phs,Frequencies,L):
    c = 299792458
    n = - (c*Phs)/(2*np.pi*Frequencies*L)

    n_diff = np.empty(len(Frequencies)-1)
    for i in range(len(Frequencies)-1):
        n_diff[i] = (n[i+1]-n[i])/(Frequencies[i+1]-Frequencies[i])

    n = n[:-1]
    Frequencies = Frequencies[:-1]
    vg = c/(n+Frequencies*n_diff)
    return n,vg

if __name__ == "__main__":
    c = 299792458
    Z0 = 50
    eps = 2.354
    mu = 1
    l = 4.8
    v = wave_velocity(eps,mu)
    v0 = 20 #初始电压
    Frequencies = np.linspace(1,60,600)*1e6

    vl1 = np.empty(len(Frequencies),dtype=complex)
    Phs1 = np.empty(len(Frequencies))
    Effs1 = np.empty(len(Frequencies))
    vl2 = np.empty(len(Frequencies),dtype=complex)
    Phs2 = np.empty(len(Frequencies))
    Effs2 = np.empty(len(Frequencies))
    for i in range(len(Frequencies)):
        f = Frequencies[i]
        alpha = 1.810E-6*np.sqrt(f)
        beta = 2* np.pi * f/v
        A1 = pro_matrix(alpha,beta,Z0,l)
        Z02 = Z0/2
        A2 = pro_matrix(alpha,beta,Z02,l)
        AT1 = A2.dot(A1.dot(A2.dot(A1.dot(A2.dot(A1.dot(A2))))))#无缺陷的传播矩阵
        vl1[i],Phs1[i],Effs1[i] = cal_phs_eff(AT1,Z0,v0)

        AT2 = A2.dot(A1.dot(A2.dot(A1.dot(A1.dot(A2.dot(A1.dot(A2)))))))#有缺陷的传播矩阵
        vl2[i],Phs2[i],Effs2[i] = cal_phs_eff(AT2,Z0,v0)


    #画图
    Phs1 = np.unwrap(Phs1,period=2*np.pi)
    n1,vg1= cal_vg(Phs1,Frequencies,l*7)
    Phs2 = np.unwrap(Phs2,period=2*np.pi)
    n2,vg2 = cal_vg(Phs2,Frequencies,l*8)
    Frequencies_1 = Frequencies[:-1]

    fg1 = plt.figure()
    plt.plot(Frequencies/1e6,Effs1,linewidth=1,color='red',linestyle='-',label='无缺陷')
    plt.plot(Frequencies/1e6,Effs2,linewidth=1,color='blue',linestyle='--',label='有缺陷')
    plt.xlabel("频率(MHz)")
    plt.ylabel("传输效率")
    plt.legend(loc='best',frameon=False)
    
    fg2 = plt.figure()
    plt.plot(Frequencies_1/1e6,n1,linewidth=1,color='red',linestyle='-',label='无缺陷')
    plt.plot(Frequencies_1/1e6,n2,linewidth=1,color='blue',linestyle='--',label='有缺陷')
    plt.xlabel("频率(MHz)")
    plt.ylabel("折射率n")
    plt.legend(loc='best',frameon=False)
    
    fg3 = plt.figure()
    plt.plot(Frequencies_1/1e6,vg1/c,linewidth=1,color='red',linestyle='-',label='无缺陷')
    plt.plot(Frequencies_1/1e6,vg2/c,linewidth=1,color='blue',linestyle='--',label='有缺陷')
    plt.xlabel("频率(MHz)")
    plt.ylabel("群速度Vg/c")
    plt.legend(loc='best',frameon=False)
    
    fg1.savefig("传输效率.pdf")
    fg2.savefig("折射率n.pdf")
    fg3.savefig("群速度.pdf")