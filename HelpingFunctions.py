

#Takes an integer and convert it to an 8bit list of binaries inverted
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
##################################################################################
"""
d – decimal integers (base-10)
f – floating point display
c – character
b – binary
o – octal
x – hexadecimal with lowercase letters after 9
X – hexadecimal with uppercase letters after 9
e – exponent notation
"""
def ConvertString2(string, return_type = 'd'):
    temp = []
    for i in string:
        temp.append(format(ord(i),return_type))
    if return_type == 'd':
        for i in range(len(temp)):
            temp[i] = int(temp[i])
    elif return_type == 'b':
        dummy = []
        for i in range(len(temp)):
            if len(temp[i]) <8:
                for _ in range(8-len(temp[i])):
                    dummy.append('0')
                dumb = ''.join(dummy)
                dummy = []
                temp[i] = dumb+ temp[i]
    return temp
##################################################################################
"""
d - decimal integers (base-10)
b – binary
h – hexadecimal
"""
def Convert2String(data, data_type = 'd'):
    if data_type == 'd':
        base = 10
    elif data_type == 'h':
        base = 16
    elif data_type == 'b':
        base = 2
    if base ==10:
        return ''.join(chr(data[i]) for i in range(len(data)))
    else:
        return ''.join(chr(int(data[i],base)) for i in range(len(data)))
######################################################################################
def Int2DecimalHex(i):
    """
    This functions takes an integer (i) and convert it to hex (xy) where x & y are both integers
    """
    hex_i = hex(i)[2:]
    return [int(hex_i[0],16), int(hex_i[1],16)]
###################################################################################
def List2_2DMatrix(list_1d):
    """
    Takes a 1D list its length is an even number and convert it to a 2D matrix.
    The matrix is ordered by column.
    """
    if len(list_1d)%4 != 0:
        raise ValueError("The list length is not even")
    else:
        index = 0
        temp_list = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        
        for j in range(int(len(list_1d)/4)): #the column index
            for i in range(int(len(list_1d)/4)): # the row index
                temp_list[i][j] = list_1d[index]
                index += 1
        return temp_list
######################################################################################
def Martix2D2List1D(matrix):
    """Takes a matrix and return a list"""
    temp_list = []
    for j in range(len(matrix)):
        for i in range(len(matrix)):
            temp_list.append(matrix[i][j])
    return temp_list
######################################################################################
def Row2Col(row):
    col = [[0],[0],[0],[0]]
    for i in range(len(row)):
         col[i][0]  = row[i]
    return col
#####################################################################################
#def Col2Row(c):
#    row = [0,0,0,0]
#    for i in range(len(c)):
#        row[0][i] = c[i][0]
#    return c
#

