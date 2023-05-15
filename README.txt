See the original project instructions here: http://ai.berkeley.edu/multiagent.html

This project focuses on multiple agents acting upon a single environment. Relevant algorithms include: Reflex-based Agent, Minimax Agent, AlphaBeta, Expectimax, and an modified Evaluation function.

Commands of note:

  Reflex Agent:
                   * python pacman.py --frameTime 0 -p ReflexAgent -k 1
                   * python pacman.py --frameTime 0 -p ReflexAgent -k 2
                   * python autograder.py -q q1

  Minimax:
                   * python autograder.py -q q2
                   * python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
                   
  αβ:
                   * python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
                   * python autograder.py -q q3
                   
  Expectimax:
                   * python autograder.py -q q4
                   * python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
                   * python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 1
                   
  Evaluation Function test: 
                   * python autograder.py -q q
