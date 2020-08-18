import matplotlib.pyplot as plt
import numpy as np
import random
import time
def create_board():
    board = np.zeros((3,3), dtype=int)
    return board
def place(board, player, position):
    if board[position] == 0:
        board[position] = player
        return board
board = create_board()

def possibilities(board):
    return list(zip(*np.where(board == 0)))

def random_place(board, player):
    x = random.choice(possibilities(board))
    return x
def row_win(board, player):
    if np.any(np.all(board==player, axis=1)): 
        return True
    else:
        return False
def col_win(board, player):
    if np.any(np.all(board==player, axis=0)):
        return True
    else:
        return False
def diag_win(board, player):
    if np.all(np.diag(board)==player) or np.all(np.diag(np.fliplr(board))==player):
        # np.diag returns the diagonal of the array
        # np.fliplr rearranges columns in reverse order
        return True
    else:
        return False
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board, player) or col_win(board, player) or diag_win(board, player):
            winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner
results = []
random.seed(1)
def play_game():
    board = create_board()
    place(board, 1, random_place(board, 1))
    place(board, 2, random_place(board, 2))
    place(board, 1, random_place(board, 1))
    place(board, 2, random_place(board, 2))
    place(board, 1, random_place(board, 1))
    place(board, 2, random_place(board, 2))
    place(board, 1, random_place(board, 1))
    place(board, 2, random_place(board, 2))
    place(board, 1, random_place(board, 1))
    evaluate(board)
    if evaluate(board) == 1 or evaluate(board) ==2:
        results.append(evaluate(board))            
        x = evaluate(board)
        return x
                
    elif possibilities == 0:
        results.append(0)
        y = evaluate(board)
        return y
                
    elif evaluate(board) == -1:
        results.append(0)
        z = evaluate(board)
        return z
for i in range(1000):
    play_game()
print(results.count(1))




