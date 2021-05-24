#import numpy as np
#from pyfinite import ffield
#import AESFunctions as AES
#
#print('forward')
#print(AES.CalcForwardSubstitutionByte(0,1,return_type='d'))
#print(AES.CalcForwardSubstitutionByte(0,1,return_type='h'))
#print(AES.CalcForwardSubstitutionByte(0,1,return_type='b'))
#print('inverse')
#print(AES.CalcInverseSubstitutionByte(0,1,return_type='d'))
#print(AES.CalcInverseSubstitutionByte(0,1,return_type='h'))
#print(AES.CalcInverseSubstitutionByte(0,1,return_type='b'))
#forward_s_box = AES.ForwardSBoxGenerator()
#inverse_s_box = AES.InverseSBoxGenerator()
#ff=open("ForwardSBox.txt","w+")
#ivf= open("IverseSBox.txt","w+")
#
#for i in range(17):
#    for j in range(17):
#        ff.write(forward_s_box[i][j] + "\t")
#        ivf.write(inverse_s_box[i][j] + "\t")
#    ff.write( "\n")
#    ivf.write( "\n")
#ff.close()
#ivf.close()
#
#def ShiftRows( data):
#        temp_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#        
#        for i in range(4):
#            for j in range(4):
#                if j+i<4:
#                    temp_list[i][j] = data[i][j+i]
#                else:
#                    temp_list[i][j] = data[i][j+i-4]
#        return temp_list
#def InverseShiftRows( data):
#        temp_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
#        
#        for i in range(4):
#            for j in range(4):
#                if j-i<0:
#                    temp_list[i][j] = data[i][j-i+4]
#                else:
#                    temp_list[i][j] = data[i][j-i]
#        return temp_list
#    
#a = [[0, 1, 2, 3],
#     [4, 5, 6, 7],
#     [8, 9, 10, 11],
#     [12, 13, 14, 15]
#     ]
#b= ShiftRows(a)
#print(b)
#print(InverseShiftRows(b))
#
#GF_256=ffield.FField(8,283)
#def ForwardMixColumn( column):
#        f_mix_col_mat = np.array([[2, 3, 1, 1],
#                              [1, 2, 3, 1],
#                              [1, 1, 2, 3],
#                              [3, 1, 1, 2]
#                             ])
#        if len(column) != 4:
#            raise ValueError("Invalid column dimention")
#        else:
#            if len(column[0]) == 1:
#                c= [0,0,0,0]
#                for i in range(4):
#                    c[i]=((GF_256.Multiply(f_mix_col_mat[i][0],column[0][0]))^ \
#                         (GF_256.Multiply(f_mix_col_mat[i][1],column[1][0]))^ \
#                         (GF_256.Multiply(f_mix_col_mat[i][2],column[2][0]))^ \
#                         (GF_256.Multiply(f_mix_col_mat[i][3],column[3][0])))%256
##                C = np.matmul(f_mix_col_mat,column)
##                #print(c)
##                C = np.mod(C,256)
#                return c
#            else:
#                raise ValueError("Invalid column dim")
#def InverseMixColumn(column):
#        i_mix_col_mat = [[14, 11, 13, 9],
#                         [9, 14, 11, 13],
#                         [13, 9, 14, 11],
#                         [11, 13, 9, 14]]
#        if len(column) != 4:
#            raise ValueError("Invalid column dimention")
#        else:
#            if len(column[0]) == 1:
#                c= [0,0,0,0]
#                for i in range(4):
#                    c[i]=((GF_256.Multiply(i_mix_col_mat[i][0],column[0][0]))^ \
#                         (GF_256.Multiply(i_mix_col_mat[i][1],column[1][0]))^ \
#                         (GF_256.Multiply(i_mix_col_mat[i][2],column[2][0]))^ \
#                         (GF_256.Multiply(i_mix_col_mat[i][3],column[3][0])))%256
#                return c
#            else:
#                raise ValueError("Invalid column dim")           
#
#b=[[124],[242],[43],[171]]
#d=ForwardMixColumn(b)     
#print(d)
#print(InverseMixColumn([[117], [85], [62], [16]]))
#import random
#def GenerateKey(s):
#        key = []
#        for i in range(int(s/8)):
#            key.append(random.randint(0,255))
#        return key
#def KeyList2WordMat(key):
#    word= []
#    j=0
#    for i in range(int(len(key)/4)):
#        word.append(key[j:j+4])
#        j+=4
#    return word
#a=GenerateKey(128)
#b=GenerateKey(192)
#c=GenerateKey(256)
#print(KeyList2WordMat(a))
#print(KeyList2WordMat(b))
#print(KeyList2WordMat(c))
#words = []
#w=[]
#words.extend(a)
#words.extend(b)
#words.extend(c)
#w.extend(KeyList2WordMat(a))
#w.extend(KeyList2WordMat(b))
#w.extend(KeyList2WordMat(c))
#print(w)
##print(words)
#j=0
#for i in range(3):
#    print(w[j:j+4])
#print(len(a))
#print(a)
#b=GenerateKey(192)
#print(len(b))
#print(b)
#c=GenerateKey(256)
#print(len(c))
#print(c)
###########################################################################################
#a = 'abD @el#A ziz'
##s = ''.join(format(ord(i),'s') for i in a)
#d=[]
#f=[]
#c=[]
#b=[]
#x=[]
#X=[]
#dumby=[]
#d1 = []
#for i in a:
#    d.append(format(ord(i),'d'))
#    c.append(format(ord(i),'c'))
#    b.append(format(ord(i),'b'))
#    x.append(format(ord(i),'x'))
#    X.append(format(ord(i),'X'))

#print(d)
#print(c)
#print(b)
#print(x)
#print(X)
#b1 = []
#b2d = []
#h2d = []
#for i in range(len(a)):
#    d1.append(hex(int(d[i]))[2:])
#    if len(b[i]) <8:
#        for _ in range(8-len(b[i])):
#            dumby.append('0')
#        dumb = ''.join(dumby)
#        dumby = []
#        b1.append(dumb+ b[i])
#    else:
#        b1[i] = b[i]
#    b2d.append(int(b1[i],2))
#    
#
#print(b2d)
#print(d1)
#print(b1)
        
#    a2 =
#    a3
#    a4
#print(s)

#c = ''.join(format(ord(i),'X') for i in a)
#d = []
#for i in range(len(c)-1):
#    if i % 2 ==0:
#        d.append( chr(int(c[i] + c[i+1],16)))
#e = ''.join(i for i in d)
#
#print(a)
##print(b)
##print(d)
#print(e)