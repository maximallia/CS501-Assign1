import numpy as np

import matplotlib.pyplot as plt

import scipy

from scipy import linalg


def error_t(matrix):
    
    data_list = []
    
    inverse_m = np.linalg.inv(matrix)
    
    base = scipy.linalg.orth(matrix)
    
    temp1 = 0
    
    t_matrix = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    for i in range(len(matrix)):
        
        v = inverse_m[i]
        
        base_k = base[i]
        
        num = t_estimate(matrix, t_matrix, v, base_k)
        
        data_list.append(num)
    
    return data_list
    
    
    
def t_estimate(matrix, t_matrix, v, base_k):
    
    # formula: 2(A^T*A*v - A^T*base_k)
    
    temp1 = np.dot(t_matrix, matrix)
    
    mul1 = np.dot(temp1, v)
    
    mul2 = np.dot(t_matrix, base_k)
    
    result = 2* (mul1 - mul2)
    
    total = 0
    
    for i in range(len( result)):
        
        total += result[i]
    
    return total
    
def m_gradient(v_t, idx, a, base, t_matrix, matrix):
    
    
    b = base[idx]
    
    mul1 = np.dot(t_matrix, b)
    
    temp2 = np.dot(matrix, v_t)
    
    mul2 = np.dot(t_matrix, temp2)
    
    last_m = mul2 - mul1
    
    result =a* last_m

    temp = v_t - result
    
    single = 0
    
    for i in range(len(temp)):
        single += temp[i]
    
    
        
    return single

#3x3
test_list = np.array([[4, 5, 6], [8, 1, 10], [7, 12, 5]])

#2X2
list2 = np.array([[2,3],[7,8]])

# 4X4
list3 = np.array([[4, 5, 6, 7,7], [8, 1, 10, 3,6], [7, 12, 5, 8,1], [1,2,3,4,8], [1,2,3,4,5]])

in_list = np.linalg.inv(test_list)

in_2 = np.linalg.inv(list2)

#transpose matrix
trans_list = [[test_list[j][i] for j in range(len(test_list))] for i in range(len(test_list[0]))]

t_2 = [[list2[j][i] for j in range(len(list2))] for i in range(len(list2[0]))]


# printing original list
print("The original list is : " + str(test_list))
  
# initialize K
K = 2
T = 1
t_o = 0

t_two = 2
  
# Get Kth Column of Matrix
# using list comprehension
res = [sub[K] for sub in test_list]
         
#get basis
base = scipy.linalg.orth(test_list)

base2 = scipy.linalg.orth(list2)



a_o = 0.1
a_two = 0.01
a_three = 0.001 
a_five = 0.0000001

idx=0
idx2= 1
idx3 = 2

v_t = [sub[idx2] for sub in in_list]

v_o = [sub[idx] for sub in in_list]

v_three = [sub[idx3] for sub in in_list]

v2_t = [sub[idx2] for sub in in_2]

v2_o = [sub[idx] for sub in in_2]

print('v0: ', v_o)

v_t_2 = m_gradient(v_o, idx, a_two, base, trans_list, test_list)

v_t_1 = m_gradient(v_o, idx, a_o, base, trans_list, test_list)

v_t_3 = m_gradient(v_o, idx, a_three, base, trans_list, test_list)

v_t_22 = m_gradient(v_t, idx2, a_two, base, trans_list, test_list)

v_t_12 = m_gradient(v_t, idx2, a_o, base, trans_list, test_list)

v_t_32 = m_gradient(v_t, idx2, a_three, base, trans_list, test_list)


#print('v2_t: ', v2_o)

v2_t_2 = m_gradient(v2_o, idx, a_five, base2, t_2, list2)

v2_t_1 = m_gradient(v2_t, idx2, a_five, base2, t_2, list2)


print('gradient update 1: ', v_t_1)

print('gradient update 2: ', v_t_2)

print('gradient 2 update 3: ', v_t_3)

print('gradient 2 update 1: ', v_t_12)

print('gradient 2 update 2: ', v_t_22)

print('gradient 2 update 3: ', v_t_32)


print('No2 gradient update 1: ', v2_t_1)

print('No2 gradient update 2: ', v2_t_2)



# part D

data_list1 = error_t(list3)

print('t data 1: ', data_list1)


data_list2 = error_t(list2)

print('t data 2: ', data_list2)

data_list3 = error_t(test_list)

print('t data 3: ', data_list3)




#print(output)

                
