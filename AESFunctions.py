from pyfinite import genericmatrix
from pyfinite import ffield 
import numpy as np

#################################################################################

"""Convert an integer into an 8bit binary inverted list so that the bit number i has an index i in the return list"""
def ConvertTo8BitsInverted(integer):
    b=[char for char in bin(integer)[2:]]
    
    c=[0,0,0,0,0,0,0,0]
    for i in range(len(b)):
        if i<len(b):
            c[i]=int(b[len(b)-1-i])
        if i>len(b):
            c[i]=0
    return c

################################################################################
"""Convert an integer into an 8bit binary list """
def ConvertTo8Bits(integer):
    b=[char for char in bin(integer)[2:]]
    
    a=['0','0','0','0','0','0','0','0']
    c = a[:8-len(b)] +b[:]
    for i in range(8):
        c[i]=int(c[i])
    return c
################################################################################
"""Convert binary number in an inverted list to a integer decimal number"""
def BitsListToDecimal(bits_list):
     a=' '
     
     for i in range(len(bits_list)):
         a=a+str(bits_list[len(bits_list)-1-i][0])
     a=a.strip()
     return int(a,2)
 
    
#####################################################################

XOR = lambda x,y : x^y
AND = lambda x,y : x&y
DIV = lambda x,y : x


###############################################################################
"""
takes a byte in hex and return value of the substitution
return_type-> (b) or (d) or (h) 
"""
def CalcForwardSubstitutionByte(x=7,y=8,return_type='b'):
    GF_256=ffield.FField(8,283) #Create a GF(2^8) with irrducible polynomial 100011011
    affine_mapping_matrix = genericmatrix.GenericMatrix(size=(8,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_vector = genericmatrix.GenericMatrix(size=(1,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_input  = genericmatrix.GenericMatrix(size=(1,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_matrix.SetRow(0,[1, 0, 0, 0, 1, 1, 1, 1])
    affine_mapping_matrix.SetRow(1,[1, 1, 0, 0, 0, 1, 1, 1])
    affine_mapping_matrix.SetRow(2,[1, 1, 1, 0, 0, 0, 1, 1])
    affine_mapping_matrix.SetRow(3,[1, 1, 1, 1, 0, 0, 0, 1])
    affine_mapping_matrix.SetRow(4,[1, 1, 1, 1, 1, 0, 0, 0])
    affine_mapping_matrix.SetRow(5,[0, 1, 1, 1, 1, 1, 0, 0])
    affine_mapping_matrix.SetRow(6,[0, 0, 1, 1, 1, 1, 1, 0])
    affine_mapping_matrix.SetRow(7,[0, 0, 0, 1, 1, 1, 1, 1])
    affine_mapping_vector.SetRow(0,[1, 1, 0, 0, 0, 1, 1, 0]) #[1, 1, 0, 0, 0, 1, 1, 0]
    affine_mapping_vector.Transpose()
   
    hex_x=hex(x)[2:]
    hex_y=hex(y)[2:]
    hex_xy_=hex_x+hex_y
    if x==0 and y==0:
        inverse_xy=0
    else:
        inverse_xy=GF_256.DoInverseForBigField(int(hex_xy_,16))
    temp =ConvertTo8BitsInverted(inverse_xy)
    affine_mapping_input.SetRow(0,temp)
    affine_mapping_input.Transpose()
    affine_mapping_output = affine_mapping_matrix * affine_mapping_input +affine_mapping_vector
    a=affine_mapping_output.data
    b=BitsListToDecimal(a)
    if return_type == 'b':
        return ConvertTo8Bits(b)
    elif return_type == 'd':
        return b;
    elif return_type == 'h':
        return hex(b)


###############################################################################
"""Takes xy byte x & y are both are integers"""
def CalcInverseSubstitutionByte(x,y,return_type='b'):
    GF_256=ffield.FField(8,283) #Create a GF(2^8) with irrducible polynomial 100011011
    affine_mapping_matrix = genericmatrix.GenericMatrix(size=(8,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_vector = genericmatrix.GenericMatrix(size=(1,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_input  = genericmatrix.GenericMatrix(size=(1,8),zeroElement=0,identityElement=1,add=XOR,mul=AND,sub=XOR,div=DIV)
    affine_mapping_matrix.SetRow(0,[0, 1, 0, 1, 0, 0, 1, 0])
    affine_mapping_matrix.SetRow(1,[0, 0, 1, 0, 1, 0, 0, 1])
    affine_mapping_matrix.SetRow(2,[1, 0, 0, 1, 0, 1, 0, 0])
    affine_mapping_matrix.SetRow(3,[0, 1, 0, 0, 1, 0, 1, 0])
    affine_mapping_matrix.SetRow(4,[0, 0, 1, 0, 0, 1, 0, 1])
    affine_mapping_matrix.SetRow(5,[1, 0, 0, 1, 0, 0, 1, 0])
    affine_mapping_matrix.SetRow(6,[0, 1, 0, 0, 1, 0, 0, 1])
    affine_mapping_matrix.SetRow(7,[1, 0, 1, 0, 0, 1, 0, 0])
    affine_mapping_vector.SetRow(0,[1, 0, 1, 0, 0, 0, 0, 0])  # I inverted it according to this online source: https://cryptography.fandom.com/wiki/Rijndael_S-box 
    affine_mapping_vector.Transpose()
    
    hex_x=hex(x)[2:]
    hex_y=hex(y)[2:]
    hex_xy_ = hex_y+ hex_x #I had to inverse the x and y in the table which also makes no sense  
    bin_xy_ = ConvertTo8BitsInverted(int(hex_xy_,16))
    affine_mapping_input.SetRow(0,bin_xy_)
    affine_mapping_input.Transpose()
    
    affine_mapping_output = affine_mapping_matrix * affine_mapping_input +affine_mapping_vector
    b_prime=affine_mapping_output.data
    int_b_prime = BitsListToDecimal(b_prime)
    
    int_A=GF_256.DoInverseForBigField(int_b_prime)
    
    if return_type== 'd':
        return int_A
    elif return_type == 'b':
        return ConvertTo8Bits(int_A)
    elif return_type == 'h':
        return hex(int_A)

###############################################################################
        
def ForwardSBoxGenerator():
    a= np.zeros((17,17))
    s_box = a.astype(str)
    for x in range(16):
        row_index=x+1
        for y in range(16):
            hex_x=hex(x)[2:]
            hex_y=hex(y)[2:]
            col_index=y+1
            s_box[row_index][col_index]=CalcForwardSubstitutionByte(x,y,return_type='h')
            if x==1 or y==1:  #Intialise the table with its headers
                s_box[0][col_index] = hex_y
                s_box[row_index][0] = hex_x
        
    return s_box



######################################################
def InverseSBoxGenerator():
    a= np.zeros((17,17))
    s_box = a.astype(str)
    for x in range(16):
        row_index=x+1
        for y in range(16):
            hex_x=hex(x)[2:]
            hex_y=hex(y)[2:]
            col_index=y+1
            s_box[row_index][col_index]=CalcInverseSubstitutionByte(x,y,return_type='h')
            if x==1 or y==1:  #Intialise the table with its headers
                s_box[0][col_index] = hex_y
                s_box[row_index][0] = hex_x
        
    return s_box
########################################################
def 



