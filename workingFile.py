from copy import deepcopy
import numpy as np
import random
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, transpile, assemble
from qiskit.providers import backend
from qiskit.aqua.components.optimizers import COBYLA

#
CAT = "c"
DOG = "d"
MOUSE = "m"
EMPTY = "emp"

# gridWorld = [[MOUSE, EMPTY, DOG],
#            [EMPTY, EMPTY, EMPTY],
#            [EMPTY, EMPTY, CAT]]

# actions:
UP = "00"
DOWN = "01"
LEFT = "10"
RIGHT = "11"

# helleo i change the main

ACTIONS = [UP, DOWN, LEFT, RIGHT]

class State:
    def __init__(self, catP):
        self.row = catP[0]
        self.column = catP[1]
        self.catP = catP

    def __eq__(self, other):
        return isinstance(other, State) and self.row == other.row and self.column == other.column and self.catP == other.catP

    def __hash__(self):
        return hash(str(self.catP))

    def __str__(self):
        return f"State(cat_pos={self.catP})"

class GridWorld:
    def __init__(self, s, catP, mouseP):
        self.numRows = s[0]
        self.numColumns = s[1]
        self.catP = catP
        self.mouseP = mouseP
        # self.dogP = dogP
        assert(not self.compaireList(self.catP, self.mouseP))
    
    def getItem(self, p):
        if p[0]>=self.numRows or p[0]<0:
            return None
        if p[1]>=self.numColumns or p[1]<0:
            return None
        if self.compaireList(p, catP):
            return CAT
        elif self.compaireList(p, mouseP):
            return MOUSE
        # elif self.compaireList(p, DOG):
        #     return DOG
        else:
            return EMPTY

    def compaireList(self, l1,l2):
        for i, j in zip(l1, l2):
            if i!=j:
                return False
        return True

    def getNumRows(self):
        return self.numRows

    def getNumColumns(self):
        return self.numColumns

    def getMouse(self):
        return self.mouse
    
    def getCatP(self):
        return self.catP

    def setCatP(self, p):
        self.catP = p
    
    def initCatState(self):
        # init cat position
        catP = [random.randint(0, self.getNumRows()), random.randint(0, self.getNumColumns())]
        while self.getItem(catP) != EMPTY and self.getItem(catP) != CAT:
            catP = [random.randint(0, self.getNumRows()), random.randint(0, self.getNumColumns())]
        self.setCatP(catP)
        return State(catP)
    
    def show(self):
        output = ""
        for i in range(self.numRows):
            for j in range(self.numColumns):
                if self.compaireList([i,j], self.catP):
                    output += CAT + " "
                elif self.compaireList([i,j], self.mouseP):
                    output += MOUSE + " "
                else:
                    output += EMPTY + " "
            output += "\n"
        print(output)

# agent: cat
class Cat:
    def __init__(self, gridWorld:GridWorld, qNet: QNet, training=True, eps = 0.2):
        self.eps = eps
        self.gw = gridWorld
        self.training = training
        self.qNet = qNet

        # result: ret = optimizer.optimize() 
        # self.rets = {(0,0):ret, (0,1):ret2,...}
        
        # we have 9 circuits here in qcs TODO: maybe a random, need try, need solve!!!
        self.state = None

    def newPosition(self, state, action):
            p = deepcopy(state.catP)
            if action == UP:
                p[0] = max(0, p[0] - 1)
            elif action == DOWN:
                p[0] = min(self.gw.getNumRows() - 1, p[0]+1)
            elif action == LEFT:
                p[1] = max(0, p[1] - 1)
            elif action == RIGHT:
                p[1] = min(self.gw.getNumColumns() - 1, p[1] + 1)
            else:
                raise ValueError(f"Unkown action {action}")
            return p

    def getReward(self, p):
        grid = self.gw.getItem(p)
        if grid == DOG:
            reward = -100
            end = True
            self.gw.setCatP(p)
        elif grid == MOUSE:
            reward = 100
            end = True
            self.gw.setCatP(p)
        elif grid == EMPTY:
            reward = -1
            end = False
            self.gw.setCatP(p)
        elif grid == CAT:
            reward = -2 # (maybe less than reward of empty)
            end = False
        else:
            raise ValueError(f"Unknown grid item {grid}")
        return reward, end

    def act(self, state, action):
        p = self.newPosition(state, action)
        reward, end = self.getReward(p)
        return deepcopy[p], reward, end
    
    def setTraining(self, training):
        self.Training = training

class QNet:
    
    def __init__(self, qTable, gridWorld:GridWorld, params, alpha, gamma):
        
        self.params = params # inital parameters are the same for all qNetwork
        self.gw = gridWorld
        self.qt = qTable
        self.backend = Aer.get_backend("qasm_simulator")
        self.NUM_SHOTS = 1000 # number of measurements 
        self.optimizer = COBYLA(maxiter=500, tol=0.0001) # off the shelf
        self.gamma = gamma
        self.alpha = alpha
        
        self.qcs = None # all qubits
        self.rets = None # resulting parameters after optimization for all points in the grid
        
        self.qc = None #current state
        self.state = None
        
        qcs = {}
        def qcMaker(params):
            qr = QuantumRegister(2, name="q")
            cr = ClassicalRegister(2, name="c")
            qc = QuantumCircuit(qr, cr)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            qc.cx(qr[0], qr[1])
            qc.measure(qr, cr)
            return qc

        for i in range(self.gw.getNumRows()):
            for j in range(self.gw.getNumColumns):
                qc = qcMaker(params)
                qcs[i, j] = qc 
    
        self.qcs = self.initQC(params)
    
    def selectAction(self, state, training):
        if random.uniform(0, 1) < self.eps:
            return random.choice(ACTIONS)
        else:
            if training:
                qc = self.qcs[state.row, state.column]
                self.state = state
                self.updateCircuit(state)
            return np.argmax(self.qt[state])
        
    def lossFunction(self, params):
        state = self.state
        qc = self.qc
        t_qc = transpile(qc, self.backend)
        job = assemble(t_qc, shots=self.NUM_SHOTS)
        rlt = self.backend.run(job).result()
        counts = rlt.get_counts(qc)
        action = max(counts, key = counts.get)
        nextPosition = self.newPosition(state, action) # handle the 
        reward, _ = self.getReward(nextPosition)
        # update q-table(but not very sure, update only for this action or for all actions)
        targetQvalue = reward + self.gamma *  np.max(self.qt[State(nextPosition)])
        if targetQvalue - self.qt[state][action] > 0:
            self.qt[state][action] += self.alpha * (targetQvalue - self.qt[state][action]) # update q-table
        return targetQvalue - self.qt[state][action]

    def updateCircuit(self, state):
        self.rets[state] = self.optimizer.optimize(num_vars=6, objective_function=self.lossFunction, initial_point=self.params)


#####################################################################################################
#new part
                    

class PetSchool:
    def __init__(self, gw:GridWorld, cat:Cat, qNetwork:QNetwork, numEpisodes, maxEpisodeSteps, training=True, minAlpha = 0.02, eps = 0.2, gamma = 1.0):
        self.gw = gw
        self.cat = cat
        self.training = training
        self.NUM_EPISODES = numEpisodes
        self.MAX_EPISODE_STEPS = maxEpisodeSteps
        self.qTable = {}
        self.alphas = np.linspace(1.0, minAlpha, self.NUM_EPISODES)
        self.gamma = gamma
        self.eps = eps
        self.ACTIONS = [UP, DOWN, LEFT, RIGHT]

    def train(self):
        counter = 0
        for e in range(self.NUM_EPISODES): #  episode: a rund for agent
            state = self.gw.initCatState()
            total_reward  = 0
            alpha = self.alphas[e]
            step = 0
            end = False
            while(step < self.MAX_EPISODE_STEPS and not end): # step: a time step for agent
                action = self.cat.qNet.selectAction(state, self.training)
                newPosition, reward, end = cat.act(state, action)
                total_reward += reward
                if self.training:
                    self.cat.qNet.updateNetwork(alpha, self.gamma, self.eps)
                step += 1
                counter += 1
        print("step: ", step, "counter: ", counter)

    def show(self):
        self.cat.setTraining(False)
        showResult(self.cat.qt)
        pass # TODO

    def initqTable(self, ACTIONS, size):
        d = {}
        for i in range(size[0]):
            for j in range(size[1]):
                d[i,j] = np.zeros(len(ACTIONS))
        return d
        
    def mouseMove(p,oldPos): # goal (mouse) moves randomly with prob p every time the cat moves
        side = 2 # Number of cells per side of the grid
        if np.random.random() < p:
            n = np.random.random()
            if n < 0.25:
                newPos = (max(0, oldPos[0]-1),oldPos[1])
            elif n < 0.5:
                newPos = (min(side - 1, oldPos[0]+1),oldPos[1])
            elif n < 0.75:
                newPos = (oldPos[0],max(0, oldPos[1]-1))
            else:
                newPos = (oldPos[0],min(side - 1, oldPos[1]+1))
        else:
            newPos = oldPos
        return newPos

gridSize = [3, 3]
catP = [gridSize[0]-1, gridSize[0]-1]
mouseP = [0, 0]
EPS = 10
MAX_EPS_STEP = 10

# initGridWorld
gridWorld = GridWorld(gridSize, catP=catP, mouseP=mouseP)

def initqTable(ACTIONS, size):
    d = {}
    for i in range(size[0]):
        for j in range(size[1]):
            d[i,j] = np.zeros(len(ACTIONS))
    return d

qt = initqTable(ACTIONS, gridSize)
cat = Cat(gridWorld)
petSchool = PetSchool(gridWorld, cat, EPS, MAX_EPS_STEP)
petSchool.train()

####################################################################################################



