from Constants import *

class Agent:
    def __init__(self, x, y, map):
        """
        x,y -- is the initial position of the agent
        """
        self.x = x
        self.y = y
        self.map = map
    
    def updatePosition(self, act):
        if act == LEFT:
            self.x -= 1
        if act == RIGHT:
            self.x += 1
        if act == UP:
            self.y += 1
        if act == DOWN:
            self.y -= 1
    
    def getAllowedActs(self):
        acts = []
        if self.map[self.x-1][self.y] != WALL:
            acts.append(LEFT)
        if self.map[self.x+1][self.y] != WALL:
            acts.append(RIGHT)
        if self.map[self.x][self.y+1] != WALL:
            acts.append(UP)
        if self.map[self.x][self.y-1] != WALL:
            acts.append(DOWN)
        return acts
    
class Pacman(Agent):
    def __init__(self, x, y, map):
        Agent.__init__(self, x, y, map)
        self.type = 'O'


class PatrolGhost(Agent):
    def __init__(self, x, y, map, k=4):
        '''
        The ghost patrols for k steps right, then k steps left, and so on. 
        '''
        Agent.__init__(self, x, y, map)
        self.dir = True #start by walking right 
        self.counter = k
        self.k = k 
        self.type = 'P'
        
    def getAct(self, pacman):
        if self.counter > 0:
            self.counter -= 1
            return RIGHT if self.dir else LEFT
        else:
            self.dir = not self.dir
            self.counter = self.k
            return self.getAct(pacman)
         
class AdversarialGhost(Agent):
    def __init__(self, x, y, map, k=10, l=1):
        '''
        If the pacman is within k steps, it walks a step towards it
        The ghost is allowed to take one step every l time steps to slow it down.
        '''
        Agent.__init__(self, x, y, map)
        self.l = l 
        self.k = k 
        self.counter = 0
        self.type = 'A'
        
    def getAct(self, pacman):
        if abs(self.x - pacman.x) + abs(self.y - pacman.y) <= self.k:
            #the ghost is only allowed to move when the counter = 0
            if self.counter > 0:
                self.counter -= 1
                return NOOP
            
            if pacman.y != self.y:
                act = UP if pacman.y > self.y else DOWN
                if act in self.getAllowedActs():
                    self.counter = self.l
                    return act 
            
            act = RIGHT if pacman.x > self.x else LEFT
            if act in self.getAllowedActs():
                self.counter = self.l
                return act 
            else:
                return NOOP
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
