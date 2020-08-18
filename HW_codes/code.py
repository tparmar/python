multiples_17 = []
multiples_11 = []
answers= []
for i in list(range(1000)):
    number = 17 * i
    multiples_17.append(number)
for n in list(range(1000)):
    number = 11 * n
    multiples_11.append(number)
for multiple in multiples_11:
    for t in range(len(multiples_17)):
        if multiple - multiples_17[t] == 1:
            answers.append((multiple/11, multiples_17[t]/17))
print(len(answers))


