# MFH2021-QRL-Cat
This is project named "**training a cat to catch a mouse**" for **Qiskit Fall Fest MUNICH 2021**.
## IDEA PITCH
To solve a grid world environment task using reinforcement learning by variational quantum circuit.
## IDEA DESCRIPTION
What is your idea or solution about?

Deep Q-learning is prevalent for reinforcement learning tasks. How about a deep Q-learning by using the variational quantum circuit. The idea of this project is to create a quantum circuit with variational gates(e.g. Phasegate with parameter) and CX-Gates(for entanglement) and to train the parameters of variational gates for the convergent expectation of Q-value. This idea is not about speedup but about quantum variational algorithms.

**Goal:**

Training a cat (single agent) to catch a mouse in a grid environment(e.g. 3*3, 4*4)

# Implementation
There are two implementation under different assumptions.

## CatHiddenMouse
Under assumption, the mouse is hidden somewhere and do not move, so the cat can not observe where the mouse is and the cat can only go straight. The task is to training the cat find the hidden mouse.

## CatMovingMouse
Under assumption, the mouse does not hid himself, but can stay or move at any time step. The cat can observe the position of the mouse at any time and the cat can go not only straight but also diagonal.
