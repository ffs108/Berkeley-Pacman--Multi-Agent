# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    #TODO
    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        cur_dist = 0 # will be distance from pacman to food pellet
        min_dist = float('inf') # arbitrarily a large number at first - then in loop the smallest distance from pacman to food - kinda like corners prob
        negative_enforcement = -float('inf') # arbitrarily low number to be returned for an unwanted action by agent
        cur_food = currentGameState.getFood().asList() # the food pellets in the current game state as an array
        for pellet in cur_food:
            cur_dist = util.manhattanDistance(newPos, pellet)
            if cur_dist < min_dist:
                min_dist = -cur_dist #has to be proportionally inverse for this to work otherwise will run for a long time with neg. score
        for state in newGhostStates:
            # print("This is state pos at 0" + str(state.getPosition()))
            # print("This is the new pacman pos" + str(newPos))
            # being checked below: first if there is a ghost nearby and if that ghost is not 'scared'
            # if this check passes we want to negatively enforce this and tell pacman by passing our neg_enf variable
            if state.getPosition() == newPos and state.scaredTimer == 0:
                return negative_enforcement
        # this does two things overall benefits the score results since we negatively enforce any period of stops pacman makes
        # and was the difference between losing outright at times
        if action == 'Stop':
            return negative_enforcement
        return min_dist #at this point this should be a relatively high number given that we made the best eval possible


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    #TODO
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"
        #minimax is going to be recursive leading to repeated calls
        #note agent 0 is always pacman

        #note for the spec hint -- pacman runs towards ghost because it assumes that dying immediately
        #is better than losing pts in a survival method hence the rush -- this is not looking at the bigger pic tho
        agentNum = gameState.getNumAgents()
        return self.minimax(0, 0, agentNum, gameState)[0]
        # we only care about the action since the score was only used to determine the best action
        # this would be the utility val in the book
 
    def minimax(self, agent, cur_depth, agentNum, gameState):
        if agent >= agentNum: # reset the agent whose turn it is add to depth
            agent = 0
            cur_depth += 1
        if gameState.isWin() or gameState.isLose() or cur_depth == self.depth: #at end return the eval -- ending edgecase
            return None, self.evaluationFunction(gameState)
        cur_optimal = None; best_action = None #cur_optimal is the score and will be the driving factor in choosing paths
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            # this would be the FUNCTION MAX or the FUNCTION MIN calls from the book algorithm
            # what we really care about to do what either of those calls would do is just check who we are currently
            # recursive call 
            resulting_score = self.minimax(agent + 1, cur_depth, agentNum, successor)[1] # depth change adv agent -- dont care about action rn
            if agent == 0:  #pacman -- higher scorer == optimal plays
                if cur_optimal is None or resulting_score > cur_optimal:
                    cur_optimal = resulting_score
                    best_action = action
            else:           #ghosts -- are also playing as optimally as they can ie lessen pacman score
                if cur_optimal is None or resulting_score < cur_optimal:
                    # in this case once we know we are not pacman we have to still play optimally, this being that pacmans score gets to
                    # be the worse than what its optimal state can be since as a ghost we kinda are trying to make pacman lose/bad score
                    cur_optimal = resulting_score
                    best_action = action
        return best_action, cur_optimal  # Return what was found to be the best action and best score 



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    #TODO
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        alpha = -float('inf')
        beta = float('inf')
        return self.alphabeta(0, 0, agentNum, gameState, alpha, beta)[0]
        # we only care about the action since the score was only used to determine the best action
        # this would be the utility val in the book
    
    def alphabeta(self, agent, cur_depth, agentNum, gameState, alpha, beta):
        if agent >= agentNum: # reset the agent whose turn it is add to depth
            agent = 0
            cur_depth += 1
        if gameState.isWin() or gameState.isLose() or cur_depth == self.depth: #at end return the eval -- ending edgecase
            return None, self.evaluationFunction(gameState)
        cur_optimal = None; best_action = None #cur_optimal is the score and will be the driving factor in choosing paths
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            # this is the exact same as the minimax recursive function that i sandwiched earlier but had to also
            # include the alpha beta components
            # recursive call 
            resulting_score = self.alphabeta(agent + 1, cur_depth, agentNum, successor, alpha, beta)[1] # depth change adv agent -- dont care about action rn
            if agent == 0:  #pacman
                if cur_optimal is None or resulting_score > cur_optimal:
                    cur_optimal = resulting_score
                    best_action = action
                alpha = max(alpha, resulting_score) # alpha = MAX(α, util val)
            else:           #ghosts
                if cur_optimal is None or resulting_score < cur_optimal:
                    cur_optimal = resulting_score
                    best_action = action
                beta = min(beta, resulting_score) # beta = MIN(β, util val)
            # my prune although the book had the pruning happening in both recursive func. I went with the single prune since logically as
            # it worked with the application... at least from the autograder not sure if this circumvented some logic that αβ should have to 
            # be general
            if alpha > beta:
                break
        return best_action, cur_optimal

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    #TODO
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        "*** YOUR CODE HERE ***"
        agentNum = gameState.getNumAgents()
        return self.expectimax(0, 0, agentNum, gameState)[0] #we only care about the action since the score was only used to determine the best action
    
    def expectimax(self, agent, cur_depth, agentNum, gameState):
        if agent >= agentNum: # reset the agent whose turn it is add to depth
            agent = 0
            cur_depth += 1
        if gameState.isWin() or gameState.isLose() or cur_depth == self.depth: #at end return the eval -- ending edgecase
            return None, self.evaluationFunction(gameState)
        cur_optimal = None; best_action = None #cur_optimal is the score and will be the driving factor in choosing paths
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            # recursive call 
            resulting_score = self.expectimax(agent + 1, cur_depth, agentNum, successor)[1] # depth change adv agent -- dont care about action rn
            if agent == 0:  #pacman -- stays the same as αβ and minimax
                if cur_optimal is None or resulting_score > cur_optimal:
                    cur_optimal = resulting_score
                    best_action = action
            else:           #ghosts
                # this probability distribution over moves believed they will make -- simplified how many moves they can make over the
                # amt of possible moves acciounting for the denom to not be zero
                # this is the "adversary which chooses amongst their getLegalActions uniformly at random."
                suboptimality_factor = (1/ len(gameState.getLegalActions(agent))) if len(gameState.getLegalActions(agent)) > 0 else 1
                if cur_optimal is None : #avoid having the optimal move be None type when attempting to add later
                    cur_optimal = 0
                # here we provide the factor as and let current optimal score to be influenced by this factor and the literal score
                # this way, while the optimal play can arise it wont all the time like in alphabeta and minimax -- its up to chance
                cur_optimal += resulting_score * suboptimality_factor
                best_action = action
        return best_action, cur_optimal  # Return what was found to be the best action and best score 
        # note: sometimes the ghosts seem a bit too optimal not sure if im thinking it or not but they do seem to still want to follow
        # pacman aronud even with the inclusion of the 'suboptimality_factor' idk 

#TODO
def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    foodList = currentGameState.getFood().asList()
    pacmanPos = currentGameState.getPacmanPosition()
    ghosts = currentGameState.getGhostStates()
    negative_enforcement = -float('inf') # arbitrarily low number to be returned for an unwanted action by agent
    closest_pellet = 0 # closest non eaten food
    food_factor = 0 ; ghost_prox = 0 #non lethal ghost will be ignored since they they arent worth anything
    # given all the food positions and the manhattan dist is acceptable to that pellet is the closest
    # this is practicially the same as the regular eval function at the top
    for food in foodList:
        man_dist = util.manhattanDistance(pacmanPos, food)
        if(man_dist < closest_pellet):
            closest_pellet = man_dist
    food_factor = -(1/3) * closest_pellet #this is .33333 since it is 1 of 3 factors making up the evaluation
    for state in ghosts:
        ghostPos = state.getPosition()
        ghost_man_dist = util.manhattanDistance(ghostPos, pacmanPos)
        if(state.scaredTimer <= 0): #ghost are lethal
            ghost_prox = -(1/3) * ghost_man_dist # a ghost(s) is approaching or getting further being factored
        else:
            ghost_prox = (1/3) * ghost_man_dist # the inverse of above this should factor telling the eval that no ghost is good path
    # pacman sometimes follows the ghost around for a while but my lines above ^^^ tells him not to
    # he will continue going for food if he is chased though so i dont understand whats going on for this
    # i did get it to 6/6 but the behavior is a bit sporadic on some of the runs -- noting for written assign. I did keep the 
    # 1/3 thing going after some talk with prof but it still seems arbitrary although I understand how he mentioned it was chosen as such but
    # since I remained constant with it, it could be valuable I kept using the score as it was a good indicator of the current impact of
    # the eval since as the score dropped the eval would subsequently also be affected by this
    return  ghost_prox + food_factor + scoreEvaluationFunction(currentGameState)


# Abbreviation
better = betterEvaluationFunction
