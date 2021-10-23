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

gridWorld = [[MOUSE, EMPTY, DOG],
        [MOUSE, EMPTY, EMPTY],
         [DOG, EMPTY, CAT]]

# actions:
UP = "00"
DOWN = "01"
LEFT = "10"
RIGHT = "11"

ACTIONS = [UP, DOWN, LEFT, RIGHT]

def showGridWorld():
    for row in gridWorld:
        print(" ".join(row))

showGridWorld()

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

# agent: cat
class cat:
    def __init__(self, eps, qTable, gridWorld, training):
        self.eps = eps
        self.qt = qTable
        self.gw = gridWorld
        self.params = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.backend = Aer.get_backend("qasm_simulator")
        self.NUM_SHOTS = 1000
        self.optimizer = COBYLA(maxiter=500, tol=0.0001)
        self.training = training
        self.rets = None
        self.qcs = self.initQC(ret[0])
        self.qc = None
        self.state = None
    
    def initQC(self, params):
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

        for i in range(len(self.gw)):
            for j in range(len(self.gw[i])):
                qc = qcMaker(params)
                qcs[i, j] = qc
        return qcs

    def selectAction(self, state):
        if random.uniform(0, 1) < self.eps:
            return random.choice(ACTIONS)
        else:
            if self.training:
                self.qc = self.qcs[state.row, state.column]
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
        targetQvalue = reward + gamma *  np.max(self.qt[State(nextPosition)])
        if targetQvalue - self.qt[state][action] > 0:
            self.qt[state][action] += alpha * (targetQvalue - self.qt[state][action]) # update q-table
        return targetQvalue - self.qt[state][action]

    def newPosition(self, state, action):
            p = deepcopy(state.catP)
            if action == UP:
                p[0] = max(0, p[0] - 1)
            elif action == DOWN:
                p[0] = min(len(self.gw) - 1, p[0]+1)
            elif action == LEFT:
                p[1] = max(0, p[1] - 1)
            elif action == RIGHT:
                p[1] = min(len(self.gw) - 1, p[1] + 1)
            else:
                raise ValueError(f"Unkown action {action}")
            return p

    def getReward(self, p):
        grid = self.gw[p[0]][p[1]]
        if grid == DOG:
            reward = -100
            end = True
            self.gw[p[0]][p[1]] += CAT
        elif grid == MOUSE:
            reward = 100
            end = True
            self.gw[p[0]][p[1]] += CAT
        elif grid == EMPTY:
            reward = -1
            end = False
            old = state.catP
            self.gw[old[0]][old[1]] = EMPTY
            self.gw[p[0]][p[1]] = CAT
        elif grid == CAT:
            reward = -1
            end = False
        else:
            raise ValueError(f"Unknown grid item {grid}")
        return reward, end

    def act(self, state, action):
        p = self.newPosition(state, action)
        reward, end = self.getReward(p)
        return deepcopy[p], reward, end
        
    def updateCircuit(self, state):
        self.rets[state] = self.optimizer.optimize(num_vars=6, objective_function=self.lossFunction, initial_point=self.params)
    
    def setTraining(self, training):
        self.Training = training

# quantum circuit: state->action
class qNetwork:
    pass

# experiment: petSchool
class petSchool:
    pass
# super parameters
N_STATES = 4
N_EPISODES = 20

MAX_EPISODE_STEPS = 100

MIN_ALPHA = 0.02

alphas = np.linspace(1.0, MIN_ALPHA, N_EPISODES)
gamma = 1.0
eps = 0.2

for e in range(N_EPISODES): #  episode: a rund for agent
    state = initState
    qTable = initqTable(len(ACTIONS))
    total_reward  = 0
    alpha = alphas[e]
    counter = 0
    step = 0
    while(step < MAX_EPISODE_STEPS): # step: a time step for agent
        action = selectAction(state)
        newPosition, reward, end = act(state, action)
        total_reward += reward
        updateNetwork()

agent.setTraining(False)
showResult(agent.qt)

def initqTable(ACTIONS):
    side = 2 # Number of cells per side of the grid
    d = {}
    for i in range(side):
        for j in range(side):
            d[(i,j)] = np.random.random()

    return d
    
