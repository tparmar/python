import string
alphabet = " " + string.ascii_lowercase 
d = dict.fromkeys(string.ascii_lowercase,0)
d1 = list(d.keys())
positions = {}
index = 0
for char in alphabet:
    positions[char] = index
    index += 1
message = "hi my name is caesar"
encoding_list = []
# for char in message:
#     position = positions[char]
#     encoded_position = (position + 1) % 27
#     encoding_list.append(alphabet[encoded_position])
# encoded_message = "".join(encoding_list)
def encoding(message, key = 0):
    encoding_list = []
    for char in message:
        position = positions[char]
        encoded_position = (position + key) % 27
        encoding_list.append(alphabet[encoded_position])
    encoded_string = "".join(encoding_list)
    return encoded_string
encoded_message = "ijanzaobnfajtadbftbs"
print(encoding(encoded_message, -3))