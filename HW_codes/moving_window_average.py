import random
random.seed(1) # This line fixes the value called by your function,
               # and is used for answer-checking.
R = 1000
x = []
def rand():
    return random.uniform(0,1)
for i in range(0,R):
    x.append(rand())
def moving_window_average(x, n_neighbors=5):
    n = len(x)
    width = n_neighbors*2 + 1
    x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    y = []
    for i in range(1,10):
        if i == 0:
            average = x[i]+ x[i+1] + x[i+2] + x[i+3]+ x[i+4]+ x[i+5] + x[i] + x[i]+ x[i]+ x[i]+ x[i]
            average_0 = average/3
            y.append(average_0)  
        elif i == 6:
            average_1 = x[i] + x[i-1] + x[i]
            average1 = average_1/3
            y.append(average1)
        else:
            average2 = x[i] + x[i+1]+ x[i-1]
            average_2 = average2/3
            y.append(average_2)
    sum1 = sum(y)
    return sum1

