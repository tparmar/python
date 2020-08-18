import numpy as np
divisors = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 25, 27, 30, 36, 40, 
45, 50, 54, 60, 72, 75, 90, 100, 108, 120, 135, 150, 180, 200, 216, 225, 
270, 300, 360, 450, 540, 600, 675, 900, 1080, 1350, 1800, 2700, 5400]
integers = list(range(74))
squares = []
answers = []
a = []
for i in range(len(integers)):
    if i == 0 or i == 1:
        continue
    else:
        squares.append(i**2)
for t in range(len(divisors)):
    for n in range(len(squares)):
        a.append(divisors[t]%squares[n])
    if not 0 in a:
        answers.append(divisors[t])
        a.clear()

    else:
        a.clear()

print(sum(answers))
                

            

