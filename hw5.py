#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 17:13:06 2018

@author: abhinavashriraam
"""

import Tkinter as tk
import numpy as np

# @function: creates a knights_tour game
# @param n: size of the board
def knights_tour(n): # Defines the function
    root = tk.Tk() # Initializes the game
    board = Knight_Moves(root, n) 
    root.mainloop()
    
    
#tkinter code    
class Knight_Moves(object):
    def __init__(self,parent,n = 5): # Sets default value of n to 5
        self.count = 0
        self.squares_covered = set([1,1]) # Adds the inital square to a set (so that I can tell when a user has won)
        self.rows = n 
        self.columns = n
        self.size = 20 
        self.color = "white"
        self.canvas_width = self.columns*self.size # dimensions of the canvas
        self.canvas_height = self.rows*self.size
        self.canvas = tk.Canvas(parent, width = self.canvas_width, height = self.canvas_height, bg = "white")
        self.canvas.pack()
        for i in np.linspace(0,self.canvas_width,n+1):
            self.canvas.create_line([(i,0),(i,self.canvas_height)]) # Creates a grid
        for j in np.linspace(0,self.canvas_height,n+1):
            self.canvas.create_line([(0,j),(self.canvas_width,j)])
        self.start()
    def start(self):
         x1 = 0 # Code to create the starting orange rectangle
         x2 = 20
         y1 = 20
         y2 = 0
         self.canvas.create_rectangle(x1, y1, x2, y2, fill = 'orange')
         self.position = (1,1)
         self.canvas.bind("<Button-1>", self.move)
    
    def legal_move(self): # Returns a list of possible legal moves based on the current position on the board
        (a,b) = self.position
        return [(a-2,b-1),(a-2,b+1),(a+2,b-1),(a+2,b+1),(a-1,b-2),(a-1,b+2),(a+1,b-2),(a+1,b+2)]
         
    def move(self,event):
        k = self.legal_move()
        for i in range(1,self.rows+1):
                if event.x > (i-1)*20 and event.x < i*20:
                    current_x = i
        for j in range(1,self.rows+1):
                if event.y > (j-1)*20 and event.y < j*20: # Identifies the move the user is trying to make
                    current_y = j
        if (current_x,current_y) in k: # If the move is legal
            x1 = (current_x -1)*20
            x2 = (current_x)*20
            y1 = (current_y)*20
            y2 = (current_y -1)*20
            self.squares_covered.add((current_x,current_y))
            self.canvas.create_rectangle(x1,y1,x2,y2, fill = 'orange') # Make a new orange square
            (a,b) = self.position
            self.position = (current_x,current_y)
            x1 = (a-1)*20
            x2 = (a)*20
            y1 = (b)*20
            y2 = (b-1)*20
            self.canvas.create_rectangle(x1,y1,x2,y2, fill = 'blue') # Make the old position a blue square
            
        else:
            print "Invalid Move" # If the move is not a legal move
        if len(self.squares_covered) == self.rows**2: # If the entire board has been covered (all n^2 squares)
            print "You have won! CONGRATULATIONS" # Success message
            print "Quitting in 3..2..1.."
            exit()
        
                            
    

    