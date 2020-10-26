import csv
import random as rd
import numpy as np

# tentukan lokasi file, nama file, dan inisialisasi csv
with open("dataset.csv") as f:
    reader= csv.reader(f)
    data= [r for r in reader]
    data.pop(0)
# menutup file csv
f.close()
#print(*data)
jmlh_data=len(data)
atribut=len(data[0])
klaster=4
m=2
e=0.001
max_i=100
u=[]
u_baru=[]
f=[]
v=[]
r=[]
d=[]
hasil=[]
iterasi=0


#matriks partisi awal
def hitPartisiAwal():
    for i in range(jmlh_data):
        temp=[]
        sum=0
        for j in range(klaster):
            a=rd.uniform(0, 1)
            temp.append (a)
            sum+=a
        
        for j in range(klaster):
            temp[j]=temp[j]/sum
        u.append(temp)
hitPartisiAwal()
f.append(0)


while (iterasi==0 or (f[iterasi]>e and iterasi<max_i)):
    #menghitung pusat klaster
    for i in range(klaster):
        temp=[]
        for j in range (atribut):
            temp2=[]
            temp3=[]
            for k in range(jmlh_data):
                a=(u[k][i]**m)
                temp2.append(a*float(data[k][j]))
                temp3.append(a)
            temp.append(np.sum(temp2)/np.sum(temp3))
        v.append(temp)
    
    #menghitung r
    for i in range(klaster):
        temp=[]
        for j in range (atribut):
            temp2=[]
            temp3=[]
            for k in range(jmlh_data):
                a=(u[k][i]**m)
                temp2.append(a*np.absolute(v[i][j]-(float(data[k][j]))))
                temp3.append(a)
            temp.append(np.sum(temp2)/np.sum(temp3))
        r.append(temp)
    
    #menghitung jarak
    d1=[]
    for i in range(klaster):
        temp=[]
        for j in range(jmlh_data):
            temp2=[]
            for k in range(atribut):
                temp2.append((np.absolute((float(data[j][k]))-v[i][k])-r[i][k])**2)
            temp.append(np.sqrt(np.sum(temp2)))
        d1.append(temp)
    d=np.transpose(d1)
    
    #update u
    for i in range (len(d)):
        temp=[]
        for j in range(len(d[i])):
            sum=0
            for k in range(len(d[i])):
                sum += d[i][j]**2/d[i][k]**2
            temp.append(1/sum)
        u_baru.append(temp)
    
    #fungsi objektif
    f_baru=0
    for i in range(len(u)):
        sum=0
        for j in range(len(u[i])):
            sum += (u[i][j]*d[i][j])
        f_baru+=sum
    f.append(np.absolute(f_baru-f[iterasi]))
    
    iterasi+=1
    u=u_baru
    u_baru=[]

#hasil akhir
for i in range (len(u)):
    hasil.append(np.argmax(u[i])+1)
    
print("Nilai U terakhir:\n", u)
print ("\nHasil klastering: \n", hasil)
print ("\nJumlah iterasi: ",iterasi)
print("\nNilai fungsi obejektif terakhir: ",f[iterasi])
