import string
key = input("Key:")
def caesar(key):
    key = int(key)
    counter = 0
    text = input("plaintext:")
    ciphertext = "ciphertext: "
    for i in range(len(text)):
        if text[i].isalpha() == False:
            ciphertext = ciphertext + text[i]
        if text[i].islower() == True:
            i = ord(text[i])
            addition = i + key
            if addition <= 122:
                i = chr(addition)
                ciphertext = ciphertext + i
            else:
                while counter < 10000000:
                    while i < 122:
                        i += 1
                        counter += 1
                    counter += 1
                    i = 97
                    test = i + key - counter
                    if test < 123:
                        test = chr(test)
                        ciphertext = ciphertext + test
                        counter = 0
                        break
        elif text[i].isupper() == True:
            i = ord(text[i])
            addition = i + key
            if addition <= 90:
                i = chr(addition)
                ciphertext = ciphertext + i
            else:
                while counter < 10000000:
                    while i < 90:
                        i += 1
                        counter += 1
                    counter += 1
                    i = 65
                    test = i + key - counter
                    if test < 91:
                        test = chr(test)
                        ciphertext = ciphertext + test
                        counter = 0
                        break
    return ciphertext
if key.isnumeric() == False:
    print("Key must be a number")
else:
    print(caesar(key))
#z = 122
#Z = 90
#A = 65
#a = 97