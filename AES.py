import base64
import hashlib
import random
from Crypto import Random
from Crypto.Cipher import AES
import numpy as np
from pyfinite import ffield
from pyfinite import genericmatrix
from AESKey import AESKey
import HelpingFunctions as helpfs
import math
import re


class AESCipher:
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

    def __init__(self, key_size: int, is_called_from_aes_key=False, is_test=True):
        if is_test:
            key = self.generate_key(256)
            self.bs = AES.block_size
            self.key = hashlib.sha256(key.encode()).digest()
        else:
            # key is the key list every field of it is a byte
            if key_size != 128 and key_size != 192 and key_size != 256:
                raise ValueError("Invalid key size")
            elif key_size == 128:
                self.n_rounds = 10
            elif key_size == 192:
                self.n_rounds = 12
            elif key_size == 256:
                self.n_rounds = 14
            self.key_size = key_size
            if not is_called_from_aes_key:
                self.aes_key = AESKey(self.key_size)
                self.key = self.aes_key.key
                self.words = self.aes_key.words

    # region AES blocks

    def encypt_(self, plaintext: list, plaintext_type="int"):
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
        ciphertext = []
        # check that the data doesn't  excced 1 byte each
        for i in range(0, len(plaintext) - 1):
            if plaintext[i] > 255:
                raise ValueError("plaintext  elements cannot excceed  1 byte")
        for i in range(0, math.ceil(len(plaintext) / 16)):
            plaintext_seg = plaintext[i * 16 : i * 16 - 1]
            if len(plaintext_seg < 16):
                for _ in range(len(plaintext_seg) - 16):
                    plaintext_seg.append(0)
            used_key_words = []
            if self.key_size == 128:
                used_key_words = self.words[0:3]
            elif self.key_size == 192:
                used_key_words = self.words[0:3]
            else:
                used_key_words = self.words[0:3]

            plaintext_seg = self.KeyAdd(
                plaintext_seg, self.aes_key.WordMat2KeyList(used_key_words)
            )
            for j in range(1, self.n_rounds - 1):
                if self.key_size == 128:
                    used_key_words = self.words[j * 4 : j * 4 - 3]
                elif self.key_size == 192:
                    used_key_words = self.words[j * 6 : j * 6 - 3]
                else:
                    used_key_words = self.words[j * 8 : j * 8 - 3]
                plaintext_seg = self.EncryRound(
                    plaintext_seg, self.aes_key.WordMat2KeyList(used_key_words)
                )

            if self.key_size == 128:
                used_key_words = self.words[48:51]
            elif self.key_size == 192:
                used_key_words = self.words[48:51]
            else:
                used_key_words = self.words[56:59]
            plaintext_seg = self.final_encryption_round(
                plaintext_seg, self.aes_key.WordMat2KeyList(used_key_words)
            )
            ciphertext.append(plaintext_seg)
        return ciphertext

    def decrypt_(self, ciphertext, ciphertext_type="int"):
        plaintext = []
        for i in range(0, math.ceil(len(ciphertext) / 16)):
            ciphertext_seg = ciphertext[i * 16 : i * 16 - 1]
            # if len(ciphertext_seg < 16):
            #     for _ in range(len(ciphertext_seg) - 16):
            #         ciphertext_seg.append(0)
            used_key_words = []
            if self.key_size == 128:
                used_key_words = self.words[40:43]
            elif self.key_size == 192:
                used_key_words = self.words[48:51]
            else:
                used_key_words = self.words[56:59]

            ciphertext_seg = self.first_decryption_round(
                ciphertext_seg, self.aes_key.WordMat2KeyList(used_key_words)
            )
            for j in range(self.n_rounds - 1, 1):
                if self.key_size == 128:
                    used_key_words = self.words[j * 4 : j * 4 - 3]
                elif self.key_size == 192:
                    used_key_words = self.words[j * 6 : j * 6 - 3]
                else:
                    used_key_words = self.words[j * 8 : j * 8 - 3]
                ciphertext_seg = self.EncryRound(
                    ciphertext_seg, self.aes_key.WordMat2KeyList(used_key_words)
                )

            if self.key_size == 128:
                used_key_words = self.words[0:3]
            elif self.key_size == 192:
                used_key_words = self.words[0:3]
            else:
                used_key_words = self.words[0:3]
            ciphertext_seg = self.KeyAdd(
                ciphertext_seg, self.aes_key.WordMat2KeyList(used_key_words)
            )
            plaintext.append(ciphertext_seg)
        return plaintext

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
        if len(key) != 16:
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

    def final_encryption_round(self, data, key):
        temp_sbox = []
        if len(data) != 16:
            raise ValueError("Data length is incorrect", data)
        if len(key) != 16:
            raise ValueError("Data length is incorrect", data)
        # SBox substitution layer.
        for i in data:
            a = helpfs.Int2DecimalHex(i)
            temp_sbox.append(self.CalcForwardSubstitutionByte(a[0], a[1]))
        # Shift Rows sublayer.
        temp_shiftR = helpfs.List2_2DMatrix(temp_sbox)
        temp_shiftR = self.ForwardShiftRows(temp_shiftR)
        # Key addition
        final_temp = self.KeyAdd(helpfs.Martix2D2List1D(temp_shiftR), key)
        return final_temp

    def decryption_round(self, data, key):
        temp_sbox = []
        if len(data) != 16:
            raise ValueError("Data length is incorrect", data)
        if len(key) != 16:
            raise ValueError("Data length is incorrect", data)
        # Key addition
        key_added = self.KeyAdd(data, key)
        # Inverse Mix Column sublayer
        temp_mix_col = []
        for i in range(0, 16, 4):
            temp_mix_col.extend(
                self.InverseMixColumn(helpfs.Row2Col(key_added[i : i + 4]))
            )
        # Inverse Shift Rows sublayer.
        temp_shiftR = helpfs.List2_2DMatrix(temp_mix_col)
        temp_shiftR = self.InverseShiftRows(temp_shiftR)
        data = helpfs.Martix2D2List1D(temp_shiftR)
        final_temp = []
        # Inverse SBox substitution layer.
        for i in data:
            a = helpfs.Int2DecimalHex(i)
            final_temp.append(self.CalcInverseSubstitutionByte(a[0], a[1]))
        return final_temp

    def first_decryption_round(self, data, key):
        temp_sbox = []
        if len(data) != 16:
            raise ValueError("Data length is incorrect", data)
        if len(key) != 16:
            raise ValueError("Data length is incorrect", data)
        # Key addition
        key_added = self.KeyAdd(data, key)
        # Inverse Shift Rows sublayer.
        temp_shiftR = helpfs.List2_2DMatrix(key_added)
        temp_shiftR = self.InverseShiftRows(temp_shiftR)
        data = helpfs.Martix2D2List1D(temp_shiftR)
        final_temp = []
        # Inverse SBox substitution layer.
        for i in data:
            a = helpfs.Int2DecimalHex(i)
            final_temp.append(self.CalcInverseSubstitutionByte(a[0], a[1]))
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
        multiplication = np.matmul(affine_mapping_matrix * affine_mapping_input)
        affine_mapping_output = np.add(multiplication + affine_mapping_vector)

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
                    self, x, y, return_type="h"
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

    # endregion

    # region Test code
    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        c = base64.b64encode(iv + cipher.encrypt(raw.encode()))
        enc = base64.b64decode(c)
        return c

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size :])).decode("utf-8")

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[: -ord(s[len(s) - 1 :])]

    def generate_key(self, key_size):
        return "".join(chr(random.randint(0, 0xFF)) for i in range(int(key_size / 8)))

    def generate_iv(self):
        return "".join([chr(random.randint(0, 0xFF)) for i in range(16)])

    @staticmethod
    def decode_base64(data, altchars=b"+/"):
        """Decode base64, padding being optional.

        :param data: Base64 data as an ASCII byte string
        :returns: The decoded byte string.

        """
        data = re.sub(rb"[^a-zA-Z0-9%s]+" % altchars, b"", data)  # normalize
        missing_padding = len(data) % 4
        if missing_padding:
            data += b"=" * (4 - missing_padding)
        return base64.b64decode(data, altchars)


# endregion
