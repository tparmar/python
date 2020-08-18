import random
import math
random.seed(1)
R = 1000
inside = []
def rand():
    return random.uniform(-1,1)
def distance(x, y):
    if len(x) != len(y):
        return "x and y do not have the same length!"
    else:
        square_differences = [(x[i] - y[i])**2 for i in range(len(x))]
        return math.sqrt(sum(square_differences))
def in_circle(x, origin = [0,0]):
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False
for i in range(0,R):
    inside.append(tuple([rand(), rand()]))
inside2 = []
for i in range(0,R):
    in_circle(inside[i])
    if in_circle(inside[i]) == True:
        inside2.append(True)
number = len(inside2)
print(number/1000)