import AES as aes
import random
class AESKey:
    
    __RC = [1,2,4,8,16,32,64,128,27,54] 
    def __init__(self, key_size):
        if key_size == 128:
            self.rounds = 10
            self.key_size = key_size
        elif key_size == 192:
            self.rounds = 12
            self.key_size = 192
        elif key_size == 256:
            self.rounds = 14
            self.key_size = 256
        else:
            raise ValueError("Incorrect Key size")
        self.key =self.GenerateKey()
        self.words = []
        self.aes =aes.AES(self.key)
        if self.key_size == 128:
            self.KeySchedule128()
        elif self.key_size == 192:
            self.KeySchedule192()
        elif self.key_size == 256:
            self.KeySchedule256()
            
        
     
    def GenerateKey(self):
        """
        Generate a random key of size key_size.
        returns a list of random integers from 0 to 255 with a length depeding on the key size. 
        Don't call it outside the AES class
        """
        key = []
        for i in range(int(self.key_size/8)):
            key.append(random.randint(0,255))
        return key
    
    #Key Schedule for 128-Bit Key AES
    def KeySchedule128(self):
        #round key 0
        self.words.extend(self.KeyList2WordMat(self.key))
        #from round 1 to round 10
        j=0
        for i in range(self.rounds):
            self.words.extend(self.FKeyRound128(self.words[j:j+4],i+1))
            j+4
    
    #Key Schedule for 192-Bit Key AES
    def KeySchedule192(self):
        #key schedule round 0
        self.words.extend(self.KeyList2WordMat(self.key))
        #key schedule rounds from 1 to 7
        j=0
        for i in range(1,8):
            self.words.extend(self.FKeyRound192(self.words[j:j+6],i))
            j+=6
        #the final round
        self.words.extend(self.FKeyRound128(self.words[42:46],8))
    
    #Key Schedule for 256-Bit Key AES
    def KeySchedule256(self):
        #key schedule round 0
        self.words.extend(self.KeyList2WordMat(self.key))
        #Key schedule rounds from 1 to 6
        j=0
        for i in range(1,7):
            self.words.extend(self.FKeyRound256(self.words[j:j+8],i))
            j+=8
        #The final key round 7
        self.words.extend(self.FKeyRound128(self.words[48:52],7))
        
    #forward Key round128
    def FKeyRound128(self,words,round_num):
        new_word_mat = words
        #the first output word
        temp_word = self.g(words[3],round_num)
        for i in range(4):
            new_word_mat[0][i] = words[0][i]^temp_word[i]
        #the second output word
        for i in range(4):
            new_word_mat[1][i] = words[0][i]^words[1][i]
        #the third output word
        for i in range(4):
            new_word_mat[2][i] = words[1][i]^words[2][i]
        #the fourth output word
        for i in range(4):
            new_word_mat[3][i] = words[2][i]^words[3][i]

        return new_word_mat
    
    
    #forward Key round192
    def FKeyRound192(self,words,round_num):
        new_word_mat = words
        #the first output word
        temp_word = self.g(words[5],round_num)
        for i in range(4):
            new_word_mat[0][i] = words[0][i] ^ temp_word[i]
        #the second output word 
        for i in range(4):
            new_word_mat[1][i] = words[0][i] ^ words[1][i]
        #the third output word 
        for i in range(4):
            new_word_mat[2][i] = words[1][i] ^ words[2][i]
        #the fourth output word
        for i in range(4):
            new_word_mat[3][i] = words[2][i] ^ words[3][i]
        #the fifth output word 
        for i in range(4):
            new_word_mat[4][i] = words[3][i] ^ words[4][i]
        #the sixth output word
        for i in range(4):
            new_word_mat[5][i] = words[4][i] ^ words[5][i]
        return new_word_mat
    
    #forward Key round256
    def FKeyRound256(self,words,round_num):
        new_word_mat = words
        #the first ouput word 0
        temp_word = self.g(words[7],round_num)
        for i in range(4):
            new_word_mat[0][i] = words[0][i] ^ temp_word[i]
        #the second,third and fourth ouput words 1,2,3
        for j in range(1,4):
            for i in range(4):
                new_word_mat[j][i] = words[j-1][i] ^ words[j][i]
        #the fifth ouput word 4
        temp_word = self.h(words[4])
        for i in range(4):
                new_word_mat[4][i] = words[3][i] ^ temp_word[i]
        ##the sixth,seventh and eighth ouput words 5,6,7
        for j in range(5,8):
            for i in range(4):
                new_word_mat[j][i] = words[j-1][i] ^ words[j][i]
        return new_word_mat
        
    #Function g of round number round_num
    def g(self, word,round_num):
        if len(word)!=4:
            
            raise ValueError("Wrong Word dimention it should be 4 byte list")
            
        temp_list = []
        for i in range(4):
            if i+1<4:
                temp_list[i] = word[i+1]
            else:
                temp_list[i] = word[i-3]
            #change from decimal integer to xy hex where x &y are in decimal
            temp_hex = hex(temp_list[i])
            temp_hex = temp_hex[2:]
            x = int(temp_hex[0],16)
            y = int(temp_hex[1],16)
            #s box sub
            temp_list[i] = self.aes.CalcForwardSubstitutionByte(x,y,'d')
        temp_list[0] ^= self.__RC[round_num -1]
        return temp_list
    
    #h −function
    def h(self,word):
        if len(word)!=4:
            
            raise ValueError("Wrong Word dimention it should be 4 byte list")
        temp_list = []
        for i in range(4):
            temp_hex = hex(word[i])
            temp_hex = temp_hex[2:]
            x = int(temp_hex[0],16)
            y = int(temp_hex[1],16)
            #s box sub
            temp_list[i] = self.aes.CalcForwardSubstitutionByte(x,y,'d')
        return temp_list
    ###################Helping functions#########################
    #change the key list to a word matrix where a row is a single word
    def KeyList2WordMat(self,key):
        word= []
        j=0
        for i in range(int(len(key)/4)):
            word.append(key[j:j+4])
            j+=4
        return word
    #change the word matrix to a key list 
    def WordMat2KeyList(self,word_list):
        key = []
        for i in range(len(word_list)):
            key.extend(word_list[i])
        return key
