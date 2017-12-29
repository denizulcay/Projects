#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
COMS W4701 Artificial Intelligence

An AI player for Othello. 

@author: Deniz Ulcay
"""

import random
import sys
import time
import math
import heapq
 
from othello_shared import find_lines, get_possible_moves, get_score, play_move

stateDict = {}

def compute_utility(board, color):
    
    a, b = get_score(board)
    
    if (color == 1):
        util = a - b
    else:
        util = b - a
        
    return util


############ MINIMAX ###############################

def minimax_min_node(board, color):
    
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    if(len(moves) == 0):
        return compute_utility(board, other)
    
    first = True
    
    for mouv in moves:
        
        pm = play_move(board, color, mouv[0], mouv[1])
        
        if (pm in stateDict):
            poss = stateDict[pm]
        else:
            poss = minimax_max_node(pm, other)
            stateDict[pm] = poss
        
        if first:
            best = poss
            first = False
        elif(poss < best):
            best = poss
    
    
    return best


def minimax_max_node(board, color):
    
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    if(len(moves) == 0):
        return compute_utility(board, color)
    
    first = True
    for mouv in moves:
        
        pm = play_move(board, color, mouv[0], mouv[1])
        
        if (pm in stateDict):
            poss = stateDict[pm]
        else:
            poss = minimax_min_node(pm, other)
            stateDict[pm] = poss
        
        if first:
            best = poss
            first = False
        elif(poss > best):
            best = poss
    
    return best
    
    
def select_move_minimax(board, color):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  
    """
        
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    first = True
    for mouv in moves:
        
        pm = play_move(board, color, mouv[0], mouv[1])
        
        if (pm in stateDict):
            poss = stateDict[pm]
        else:
            poss = minimax_min_node(pm, other)
            stateDict[pm] = poss
        
        if first:
            best = poss
            theMove = mouv
            first = False
        elif(poss > best):
            best = poss
            theMove = mouv
    
    return theMove

    
############ ALPHA-BETA PRUNING #####################

def alphabeta_min_node(board, color, alpha, beta, level, limit):
#def alphabeta_min_node(board, color, alpha, beta):
    
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    if(len(moves) == 0):
        return compute_utility(board, other)   
    
    h = []
    for mouv in moves:
        pm = play_move(board, color, mouv[0], mouv[1])
        u = compute_utility(pm, other)
        u = -u
        
        heapq.heappush(h, (u, mouv))
    
    v = math.inf
    
    for i in range(len(h)):
        
        mouv = heapq.heappop(h)[1]
        
        pm = play_move(board, color, mouv[0], mouv[1])

        if(level + 1 == limit):
            poss = compute_utility(pm, other)         
        elif(pm in stateDict):
            poss = stateDict[pm]
        else:
            poss = alphabeta_max_node(pm, other, alpha, beta, level + 1, limit)
            stateDict[pm] = poss
        
        v = min(v, poss)
        if(v <= alpha):
            return v
        
        beta = min(beta, v)
          
    return v
    

def alphabeta_max_node(board, color, alpha, beta, level, limit):
#def alphabeta_max_node(board, color, alpha, beta):
    
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    if(len(moves) == 0):
        return compute_utility(board, color)
    
    h = []
    for mouv in moves:
        pm = play_move(board, color, mouv[0], mouv[1])
        u = compute_utility(pm, color)
        u = -u
        
        heapq.heappush(h, (u, mouv))
    
    v = -math.inf
    
    for i in range(len(h)):
        
        mouv = heapq.heappop(h)[1]      
        pm = play_move(board, color, mouv[0], mouv[1])
        
        if(level + 1 == limit):
            poss = compute_utility(pm, color)     
        elif(pm in stateDict):
            poss = stateDict[pm]
        else:        
            poss = alphabeta_min_node(pm, other, alpha, beta, level + 1, limit)
            stateDict[pm] = poss
        
        v = max(v, poss)
        if(v >= beta):
            return v
        
        alpha = max(alpha, v)
          
    return v


def select_move_alphabeta(board, color):
    
    limit = 6
    
    if(color == 1):
        other = 2
    else:
        other = 1
        
    moves = get_possible_moves(board, color)
    
    h = []
    for mouv in moves:
        pm = play_move(board, color, mouv[0], mouv[1])
        u = compute_utility(pm, color)
        u = -u
        
        heapq.heappush(h, (u, mouv))
    
    first = True
    
    for i in range(len(h)):
        
        mouv = heapq.heappop(h)[1]
        pm = play_move(board, color, mouv[0], mouv[1])
        
        if(pm in stateDict):
            poss = stateDict[pm]
        else:         
            poss = alphabeta_min_node(pm, other, -math.inf, math.inf, 1, limit)
            stateDict[pm] = poss
        
        if first:
            best = poss
            theMove = mouv
            first = False
        elif(poss < best):
            best = poss
            theMove = mouv
    
    return theMove


####################################################
def run_ai():
    """
    This function establishes communication with the game manager. 
    It first introduces itself and receives its color. 
    Then it repeatedly receives the current score and current board state
    until the game is over. 
    """
    print("Minimax AI") # First line is the name of this AI  
    color = int(input()) # Then we read the color: 1 for dark (goes first), 
                         # 2 for light. 

    while True: # This is the main loop 
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input() 
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over. 
            print 
        else: 
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The 
                                  # squares in each row are represented by 
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)
                    
            # Select the move and send it to the manager 
            movei, movej = select_move_minimax(board, color)
            #movei, movej = select_move_alphabeta(board, color)
            print("{} {}".format(movei, movej)) 


if __name__ == "__main__":
    run_ai()
