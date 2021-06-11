import numpy as np
from pyfinite import ffield
from pyfinite import genericmatrix
import AESKey as keym
import HelpingFunctions as helpfs


class AES:
    """
    AES encryption decryption algorithm.
    The Encryption data enters as dicimal bytes.
    The data is 16 byte/ 128 bit, and is in a list of length 16 integers.
    """

    GF_256 = ffield.FField(8, 283)

    # Bitwise operation opretators
    XOR = lambda x, y: x ^ y
    AND = lambda x, y: x & y
    DIV = lambda x, y: x

    def __init__(self, key_size):
        # key is the key list every field of it is a byte
        if key_size not in (128, 192, 256):
            raise ValueError("Invalid key size")
        elif key_size == 128:
            self.n_rounds = 10
        elif key_size == 192:
            self.n_rounds = 12
        elif key_size == 256:
            self.n_rounds = 14
        self.key_size = key_size
        self.aes_key = keym.AESkey(self.key_size)
        self.key = self.aes_key.key

    def encypt_plaintext(self, plaintext, plaintext_type="int"):
        """
        This method encrypt plain  text of type plaintext_type and return the encrypted
        text.

        Keyword arguments:
        plaintext -- The plaintext that will  be  encrypted
        plaintext_type -- the type of plain  text to be encrypted
        plaintext_type types:
        int -> array of integers values
        bit -> array of binary values
        str -> for string input
        hex -> array  of  hex values
        """
        pass

    def EncryRound(self, data, key):

        """
        The encryption single round.

        Keyword arguments:
        data -- a 1D list of integer data.
        key -- the  encryption key which  is  a  list of integers with length depending on  the key  size.

        """

        temp_sbox = []
        if len(data) != 16:
            raise ValueError("Data length is incorrect", data)
        # SBox substitution layer.
        for i in data:
            a = helpfs.Int2DecimalHex(i)
            temp_sbox.append(self.CalcForwardSubstitutionByte(a[0], a[1]))
        # Shift Rows sublayer.
        temp_shiftR = helpfs.List2_2DMatrix(temp_sbox)
        temp_shiftR = self.ForwardShiftRows(temp_shiftR)
        # Mix Column sublayer
        temp_mixCol = helpfs.Martix2D2List1D(temp_shiftR)
        temp_mixCol1 = []
        for i in range(0, 16, 4):
            temp_mixCol1.extend(
                self.ForwardMixColumn(helpfs.Row2Col(temp_mixCol[i : i + 4]))
            )
        # Key addition
        final_temp = self.KeyAdd(temp_mixCol1, key)
        return final_temp

    def CalcForwardSubstitutionByte(self, x, y, return_type="d"):
        """
        Takes a byte in hex and return value of the substitution
        return_type-> (b) or (d) or (h)
        """
        GF_256 = ffield.FField(
            8, 283
        )  # Create a GF(2^8) with irrducible polynomial 100011011
        affine_mapping_matrix = genericmatrix.GenericMatrix(
            size=(8, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_vector = genericmatrix.GenericMatrix(
            size=(1, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_input = genericmatrix.GenericMatrix(
            size=(1, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_matrix.SetRow(0, [1, 0, 0, 0, 1, 1, 1, 1])
        affine_mapping_matrix.SetRow(1, [1, 1, 0, 0, 0, 1, 1, 1])
        affine_mapping_matrix.SetRow(2, [1, 1, 1, 0, 0, 0, 1, 1])
        affine_mapping_matrix.SetRow(3, [1, 1, 1, 1, 0, 0, 0, 1])
        affine_mapping_matrix.SetRow(4, [1, 1, 1, 1, 1, 0, 0, 0])
        affine_mapping_matrix.SetRow(5, [0, 1, 1, 1, 1, 1, 0, 0])
        affine_mapping_matrix.SetRow(6, [0, 0, 1, 1, 1, 1, 1, 0])
        affine_mapping_matrix.SetRow(7, [0, 0, 0, 1, 1, 1, 1, 1])
        affine_mapping_vector.SetRow(
            0, [1, 1, 0, 0, 0, 1, 1, 0]
        )  # [1, 1, 0, 0, 0, 1, 1, 0]
        affine_mapping_vector.Transpose()

        hex_x = hex(x)[2:]
        hex_y = hex(y)[2:]
        hex_xy_ = hex_x + hex_y
        if x == 0 and y == 0:
            inverse_xy = 0
        else:
            inverse_xy = GF_256.DoInverseForBigField(int(hex_xy_, 16))
        temp = helpfs.ConvertTo8BitsInverted(inverse_xy)
        affine_mapping_input.SetRow(0, temp)
        affine_mapping_input.Transpose()
        affine_mapping_output = (
            affine_mapping_matrix * affine_mapping_input + affine_mapping_vector
        )
        a = affine_mapping_output.data
        b = helpfs.BitsListToDecimal(a)
        if return_type == "b":
            return helpfs.ConvertTo8Bits(b)
        elif return_type == "d":
            return b
        elif return_type == "h":
            return hex(b)

    def CalcInverseSubstitutionByte(self, x, y, return_type="d"):
        """Takes xy byte x & y are both are integers"""

        GF_256 = ffield.FField(
            8, 283
        )  # Create a GF(2^8) with irrducible polynomial 100011011
        affine_mapping_matrix = genericmatrix.GenericMatrix(
            size=(8, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_vector = genericmatrix.GenericMatrix(
            size=(1, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_input = genericmatrix.GenericMatrix(
            size=(1, 8),
            zeroElement=0,
            identityElement=1,
            add=self.XOR,
            mul=self.AND,
            sub=self.XOR,
            div=self.DIV,
        )
        affine_mapping_matrix.SetRow(0, [0, 1, 0, 1, 0, 0, 1, 0])
        affine_mapping_matrix.SetRow(1, [0, 0, 1, 0, 1, 0, 0, 1])
        affine_mapping_matrix.SetRow(2, [1, 0, 0, 1, 0, 1, 0, 0])
        affine_mapping_matrix.SetRow(3, [0, 1, 0, 0, 1, 0, 1, 0])
        affine_mapping_matrix.SetRow(4, [0, 0, 1, 0, 0, 1, 0, 1])
        affine_mapping_matrix.SetRow(5, [1, 0, 0, 1, 0, 0, 1, 0])
        affine_mapping_matrix.SetRow(6, [0, 1, 0, 0, 1, 0, 0, 1])
        affine_mapping_matrix.SetRow(7, [1, 0, 1, 0, 0, 1, 0, 0])
        affine_mapping_vector.SetRow(
            0, [1, 0, 1, 0, 0, 0, 0, 0]
        )  # I inverted it according to this online source: https://cryptography.fandom.com/wiki/Rijndael_S-box
        affine_mapping_vector.Transpose()

        hex_x = hex(x)[2:]
        hex_y = hex(y)[2:]
        hex_xy_ = (
            hex_y + hex_x
        )  # I had to inverse the x and y in the table which also makes no sense
        bin_xy_ = helpfs.ConvertTo8BitsInverted(int(hex_xy_, 16))
        affine_mapping_input.SetRow(0, bin_xy_)
        affine_mapping_input.Transpose()

        affine_mapping_output = (
            affine_mapping_matrix * affine_mapping_input + affine_mapping_vector
        )
        b_prime = affine_mapping_output.data
        int_b_prime = helpfs.BitsListToDecimal(b_prime)

        int_A = GF_256.DoInverseForBigField(int_b_prime)

        if return_type == "d":
            return int_A
        elif return_type == "b":
            return helpfs.ConvertTo8Bits(int_A)
        elif return_type == "h":
            return hex(int_A)

    def ForwardSBoxGenerator(self):
        """
        Generate the Forward SBox look up table and return it as a numpy array that can be printed directly
        """
        a = np.zeros((17, 17))
        s_box = a.astype(str)
        for x in range(16):
            row_index = x + 1
            for y in range(16):
                hex_x = hex(x)[2:]
                hex_y = hex(y)[2:]
                col_index = y + 1
                s_box[row_index][col_index] = self.CalcForwardSubstitutionByte(
                    x, y, return_type="h"
                )
                if x == 1 or y == 1:  # Intialise the table with its headers
                    s_box[0][col_index] = hex_y
                    s_box[row_index][0] = hex_x

        return s_box

    def InverseSBoxGenerator(self):
        """
        Generate the Inverse SBox look up table and return it as a numpy array that can be printed directly
        """
        a = np.zeros((17, 17))
        s_box = a.astype(str)
        for x in range(16):
            row_index = x + 1
            for y in range(16):
                hex_x = hex(x)[2:]
                hex_y = hex(y)[2:]
                col_index = y + 1
                s_box[row_index][col_index] = self.CalcInverseSubstitutionByte(
                    x, y, return_type="h"
                )
                if x == 1 or y == 1:  # Intialise the table with its headers
                    s_box[0][col_index] = hex_y
                    s_box[row_index][0] = hex_x

        return s_box

    def ForwardShiftRows(self, data):
        """
        Forward Shift Rows sublayer.
        Takes the data as 4x4 2D list.
        """
        temp_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        for i in range(4):
            for j in range(4):
                if j + i < 4:
                    temp_list[i][j] = data[i][j + i]
                else:
                    temp_list[i][j] = data[i][j + i - 4]
        return temp_list

    def InverseShiftRowa(self, data):
        """
        Inverse Shift Rows sublayer.
        Takes the data as 4x4 2D list.
        """
        temp_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        for i in range(4):
            for j in range(4):
                if j - i < 0:
                    temp_list[i][j] = data[i][j - i + 4]
                else:
                    temp_list[i][j] = data[i][j - i]
        return temp_list

    #
    def ForwardMixColumn(self, column):
        """
        Forward Mix Column sublayer.
        Takes a column of the matrix that is a column list.
        Returns a 1D row list.
        """
        f_mix_col_mat = np.array(
            [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]
        )
        if len(column) != 4:
            raise ValueError("Invalid column dimention")
        else:
            if len(column[0]) == 1:
                c = [0, 0, 0, 0]
                for i in range(4):
                    c[i] = (
                        (self.GF_256.Multiply(f_mix_col_mat[i][0], column[0][0]))
                        ^ (self.GF_256.Multiply(f_mix_col_mat[i][1], column[1][0]))
                        ^ (self.GF_256.Multiply(f_mix_col_mat[i][2], column[2][0]))
                        ^ (self.GF_256.Multiply(f_mix_col_mat[i][3], column[3][0]))
                    ) % 256
                return c
            else:
                raise ValueError("Invalid column dim")

    def InverseMixColumn(self, column):
        """
        Inverse Mix Column sublayer.
        Takes a column of the matrix that is a column list.
        Returns a 1D row list.
        """
        i_mix_col_mat = [
            [14, 11, 13, 9],
            [9, 14, 11, 13],
            [13, 9, 14, 11],
            [11, 13, 9, 14],
        ]
        if len(column) != 4:
            raise ValueError("Invalid column dimention")
        else:
            if len(column[0]) == 1:
                c = [0, 0, 0, 0]
                for i in range(4):
                    c[i] = (
                        (self.GF_256.Multiply(i_mix_col_mat[i][0], column[0][0]))
                        ^ (self.GF_256.Multiply(i_mix_col_mat[i][1], column[1][0]))
                        ^ (self.GF_256.Multiply(i_mix_col_mat[i][2], column[2][0]))
                        ^ (self.GF_256.Multiply(i_mix_col_mat[i][3], column[3][0]))
                    ) % 256
                return c
            else:
                raise ValueError("Invalid column dim")

        # Key addition

    def KeyAdd(self, data, key):
        for i in range(16):
            data[i] ^= key[i]
