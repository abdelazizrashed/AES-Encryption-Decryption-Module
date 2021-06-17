from AES import AESCipher

key_size = input("Enter the key size: ")

aes_model = AESCipher(int(key_size))

key = aes_model.key

print("The encryption key: " + str(key))

message = input("Enter the message to be encrypted: ")

encrypted_message = aes_model.encrypt(message)

print("The encrypted message: " + str(encrypted_message))

print("decrypting the message now")

new_message = aes_model.decrypt(encrypted_message)

print("The decrypted message: " + str(new_message))

# plaintext = [
#     234,
#     45,
#     66,
#     77,
#     8,
#     99,
#     0,
#     19,
#     19,
#     19,
#     20,
#     29,
#     32,
#     78,
#     98,
#     23,
#     34,
#     66,
#     33,
#     44,
#     55,
#     22,
#     11,
#     243,
#     53,
#     4,
# ]

# print("The plain text to be encrypted: " + str(plaintext))

# ciphertext = aes_model.encypt_plaintext(plaintext)

# print("The ciphertext" + str(ciphertext))

# print("Now decryption the ciphertext")

# new_plaintext = aes_model.decrypt_ciphertext(ciphertext)

# print("the decrypted cipher text: " + str(new_plaintext))