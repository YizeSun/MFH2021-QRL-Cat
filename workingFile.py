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

random.seed(10)

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
                if self.compaireList([i,j], self.mouseP):
                    output += MOUSE + " "
                if not self.compaireList([i,j], self.catP) and not self.compaireList([i,j], self.mouseP):
                    output += EMPTY + " "
            output += "\n"
        print(output)

class QNet:
    
    def __init__(self, qTable, gridWorld:GridWorld, params, alpha=0.1, gamma=1.0, eps=0.2, actions=[UP, DOWN, LEFT, RIGHT]):
        
        self.params = params # inital parameters are the same for all qNetwork
        self.gw = gridWorld
        self.qt = qTable
        self.eps = eps
        self.backend = Aer.get_backend("qasm_simulator")
        self.NUM_SHOTS = 1000 # number of measurements 
        self.optimizer = COBYLA(maxiter=500, tol=0.0001) # off the shelf
        self.gamma = gamma
        self.alpha = alpha
        self.ACTIONS = actions

        self.qcs = dict() # all qubits
        self.rets = dict() # resulting parameters after optimization for all points in the grid
        
        self.qc = None #current state
        self.state = None
        
        qcs = {}
        def qcMaker(params):
            qr = QuantumRegister(2, name="q")
            cr = ClassicalRegister(2, name="c")
            qc = QuantumCircuit(qr, cr)
            qc.u3(params[0], params[1], params[2], qr[0])
            qc.u3(params[3], params[4], params[5], qr[1])
            # qc.cx(qr[0], qr[1])
            qc.measure(qr, cr)
            return qc

        for i in range(self.gw.getNumRows()):
            for j in range(self.gw.getNumColumns()):
                qc = qcMaker(params)
                qcs[i, j] = qc 
    
        self.qcs = qcs
    
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
            reward = -1000
        elif grid == MOUSE:
            reward = 100
        elif grid == EMPTY:
            reward = -1
        elif grid == CAT:
            reward = -1 # (maybe less than reward of empty)
        else:
            raise ValueError(f"Unknown grid item {grid}")
        return reward
    
    def selectAction(self, state, training):
        if random.uniform(0, 1) < self.eps:
            return random.choice(self.ACTIONS)
            # return int(random.choice(self.ACTIONS), 2)
        else:
            if training:
                self.qc = self.qcs[state.row, state.column]
                self.state = state
                self.updateCircuit()
            return self.ACTIONS[np.argmax(self.qt[self.state.catP[0], self.state.catP[1]])]
        
    def lossFunction(self, params):
        #state = self.state
        #qc = self.qc
        t_qc = transpile(self.qc, self.backend)
        job = assemble(t_qc, shots=self.NUM_SHOTS)
        rlt = self.backend.run(job).result()
        counts = rlt.get_counts(self.qc) 
        action = max(counts, key = counts.get)
        nextPosition = self.newPosition(self.state, action) # handle the 
        reward = self.getReward(nextPosition)
        # update q-table(but not very sure, update only for this action or for all actions)
        targetQvalue = reward + self.gamma *  np.max(self.qt[nextPosition[0],nextPosition[1]])
        predictedQvalue = self.calculateQvalue(action, nextPosition, reward, self.state)
        
        # update q-table
        # self.updateQtable(predictedQvalue, action)

        return targetQvalue - self.qt[self.state.catP[0],self.state.catP[1]][int(action,2)]

    def updateQtable(self, predictedQvalue, action):
        if self.qt[(self.state.catP[0],self.state.catP[1])][int(action,2)] < predictedQvalue:
            self.qt[self.state.catP[0],self.state.catP[1]][int(action,2)] = predictedQvalue

    def calculateQvalue(self, action, nextPosition, reward, state):
        targetQvalue = reward + self.gamma *  np.max(self.qt[nextPosition[0],nextPosition[1]])
        return self.qt[state.catP[0], state.catP[1]][int(action,2)] + self.alpha * (targetQvalue - self.qt[state.catP[0],state.catP[1]][int(action,2)]) # update q-table

    def updateCircuit(self):
        self.rets[self.state.catP[0],self.state.catP[1]] = self.optimizer.optimize(num_vars=6, objective_function=self.lossFunction, initial_point=self.params)

    def setAlpha(self, alpha):
        self.alpha = alpha

# agent: cat
class Cat:
    def __init__(self, qNet: QNet, training=True, eps = 0.2, actions = [UP, DOWN, LEFT, RIGHT]):
        self.eps = eps
        self.training = training
        self.qNet = qNet
        self.ACTIONS = actions

        # result: ret = optimizer.optimize() 
        # self.rets = {(0,0):ret, (0,1):ret2,...}
        
        # we have 9 circuits here in qcs TODO: maybe a random, need try, need solve!!!
        self.state = None

    def newPosition(self, state, action):
            p = deepcopy(state.catP)
            if action == UP:
                p[0] = max(0, p[0] - 1)
            elif action == DOWN:
                p[0] = min(self.qNet.gw.getNumRows() - 1, p[0] + 1)
            elif action == LEFT:
                p[1] = max(0, p[1] - 1)
            elif action == RIGHT:
                p[1] = min(self.qNet.gw.getNumColumns() - 1, p[1] + 1)
            else:
                raise ValueError(f"Unkown action {self.ACTIONS[action]}")
            return p

    def getReward(self, p):
        grid = self.qNet.gw.getItem(p)
        if grid == MOUSE:
            reward = 1000
            end = True
            self.qNet.gw.setCatP(p)
        # elif grid == DOG:
        #     reward = -100
        #     end = True
        #     self.qNet.gw.setCatP(p)
        elif grid == EMPTY:
            reward = -1
            end = False
            self.qNet.gw.setCatP(p)
        elif grid == CAT:
            reward = -1 # (maybe less than reward of empty)
            end = False
        else:
            raise ValueError(f"Unknown grid item {grid}")
        return reward, end

    def act(self, state, action):
        p = self.newPosition(state, action)
        reward, end = self.getReward(p)
        return p, reward, end
    
    def updateQtable(self, action, p, reward, state):
        pqv = self.qNet.calculateQvalue(action, p, reward, state)
        self.qNet.updateQtable(pqv, action)

    def setTraining(self, training):
        self.Training = training
                    

class PetSchool:
    def __init__(self, cat:Cat, numEpisodes, maxEpisodeSteps, training=True, minAlpha = 0.02, eps = 0.2):
        self.cat = cat
        self.training = training
        self.NUM_EPISODES = numEpisodes
        self.MAX_EPISODE_STEPS = maxEpisodeSteps
        self.alphas = np.linspace(1.0, minAlpha, self.NUM_EPISODES)
        self.eps = eps

    def train(self):
        counter = 0
        for e in range(self.NUM_EPISODES): #  episode: a rund for agent
            print("episode: ", e)
            state = self.cat.qNet.gw.initCatState()
            self.cat.qNet.setAlpha(self.alphas[e])
            total_reward  = 0
            step = 0
            end = False
            for _ in range(self.MAX_EPISODE_STEPS): # step: a time step for agent
                action = self.cat.qNet.selectAction(state, self.training)
                p, reward, end = self.cat.act(state, action)
                self.cat.updateQtable(action, p, reward, state)
                total_reward += reward
                step += 1
                counter += 1
                if end:
                    print("catch the mouse!!!")
                    print("total reward: ", total_reward, "steps: ", step)
                    break
        print("counter: ", counter)

    def show(self):
        # self.cat.setTraining(False)
        self.cat.qNet.gw.show()
        print("qTable: ", self.cat.qNet.qt)
        print("params: ", self.cat.qNet.rets )

    def initqTable(self, actions, size):
        d = {}
        for i in range(size[0]):
            for j in range(size[1]):
                d[i,j] = np.zeros(len(actions))
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
EPS = 50
MAX_EPS_STEP = 50
sizeOfParams = 6
gamma = 0.9

def initqTable(size, actions=[UP, DOWN, LEFT, RIGHT]):
    d = {}
    for i in range(size[0]):
        for j in range(size[1]):
            d[i,j] = np.zeros(len(actions))
    return d

# initGridWorld
gridWorld = GridWorld(gridSize, catP=catP, mouseP=mouseP)
initialParameters = np.zeros(sizeOfParams)
qt = initqTable(gridSize)
qNet = QNet(qt, gridWorld, initialParameters, gamma=gamma)
cat = Cat(qNet=qNet)
petSchool = PetSchool(cat, EPS, MAX_EPS_STEP)
petSchool.train()
petSchool.show()



