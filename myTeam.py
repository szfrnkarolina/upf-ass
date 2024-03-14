# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start_state = None
        self.central_home_x = None
        self.legal_positions = None
        self.dead_ends = None
        self.central_home_positions = None
        self.start = None
        self.on_defence = None

    def register_initial_state(self, game_state):
        self.start_state = game_state
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        width = game_state.data.layout.width
        height = game_state.data.layout.height

        # Store legal central home positions
        central_home_positions = []

        if self.red:
            self.central_home_x = int(width / 2 - 1)
        else:
            self.central_home_x = int(width / 2)

        for i in range(1, height - 1):
            if not game_state.has_wall(self.central_home_x, i):
                central_home_positions.append((self.central_home_x, i))

        self.central_home_positions = central_home_positions

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        if self.on_defence:
            features = self.get_defence_features(game_state, action)
            weights = self.get_defence_weights()
        else:
            features = self.get_offence_features(game_state, action)
            weights = self.get_offence_weights(game_state, action)
        return features * weights

    def get_scared_ghosts_num(self, successor):
        num = 0
        opponents = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        for opponent in opponents:
            if opponent.scared_timer != 0:
                num += 1
        return num

    def get_pacman_enemies_num(self, successor):
        num = 0
        opponents = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        for opponent in opponents:
            if opponent.is_pacman:
                num += 1
        return num

    def get_unscared_ghosts(self, successor):
        opponents = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        return [opponent for opponent in opponents if not opponent.is_pacman
                and opponent.get_position() is not None and opponent.scared_timer == 0]

    def get_scared_ghosts(self, successor):
        opponents = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        return [opponent for opponent in opponents if not opponent.is_pacman
                and opponent.get_position() is not None and opponent.scared_timer != 0]

    def get_nearest_ghost_dist(self, successor, my_pos):
        ghosts = self.get_unscared_ghosts(successor)
        if len(ghosts) > 0:
            return min([self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts])
        return None

    def get_nearest_capsule_dist(self, game_state, my_pos):
        capsules = self.get_capsules(game_state)
        if len(capsules) > 0:
            return min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
        return None

    def get_other_team_agent(self, game_state):
        for i in self.get_team(game_state):
            if i != self.index:
                return game_state.get_agent_state(i)
        return None

    def get_recent_death(self, game_state):
        previous_game_state = self.get_previous_observation()
        if previous_game_state:
            previous_food_to_eat = self.get_food(previous_game_state)
            current_food_to_eat = self.get_food(game_state)
            return len(previous_food_to_eat.as_list()) < len(current_food_to_eat.as_list())  # if there was less food to eat in the previous state that means a pacman died
        return False

    def switch_sides(self, game_state, is_pacman):
        recent_death = self.get_recent_death(game_state)
        other_agent = self.get_other_team_agent(game_state)
        if recent_death and other_agent and not other_agent.is_pacman and not is_pacman:  # i am not pacman and other agent is not pacman = we are both on our side of the maze
            my_current_pos = game_state.get_agent_state(self.index).get_position()
            other_agent_current_pos = other_agent.get_position()

            my_min_distance = min(
                [self.get_maze_distance(my_current_pos, central_pos) for central_pos in
                 self.central_home_positions])

            agent_min_distance = min(
                [self.get_maze_distance(other_agent_current_pos, central_pos) for central_pos in
                 self.central_home_positions])

            if self.on_defence:
                if agent_min_distance > 1.5 * my_min_distance:
                    print('recent death, both on their side, no one is pacman, changing to offence', self.index)
                    self.on_defence = False
            else:
                if my_min_distance >= 1.5 * agent_min_distance:
                    print('recent death, both on their side, no one is pacman, changing to defence', self.index)
                    self.on_defence = True

    def get_offence_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        is_pacman_current = game_state.get_agent_state(self.index).is_pacman
        is_pacman_successor = successor.get_agent_state(self.index).is_pacman
        my_pos = successor.get_agent_state(self.index).get_position()

        # Compute amount of food left to eat
        # -> we want to minimise the number of food left to eat = maximising the score
        food_list = self.get_food(successor).as_list()  # food that is available to eat
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food
        # -> we want to minimise the distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance_to_food = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance_to_food

        # Compute distance to ghosts (opponents), only if not scared
        # -> we want some distance between a ghost and our agent
        nearest_ghost_dist = self.get_nearest_ghost_dist(successor, my_pos)
        if is_pacman_successor and nearest_ghost_dist:
            features['distance_to_ghost'] = nearest_ghost_dist

        # Compute distance to capsule
        # -> we want to minimise distance to the nearest capsule
        nearest_capsule_dist = self.get_nearest_capsule_dist(game_state, my_pos)
        if nearest_capsule_dist:
            features['distance_to_capsules'] = nearest_capsule_dist

        # Compute distance to invaders (offensive agent should also chase invaders if they are close)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if not is_pacman_successor and len(invaders) > 0:
            closest_invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            if closest_invader_distance < 4:
                features['invader_distance'] = closest_invader_distance

        # Compute distance to your side (ie to central column of your team)
        # -> we want to minimise this distance
        num_scared_ghosts = self.get_scared_ghosts_num(successor)
        num_food_carried = game_state.get_agent_state(self.index).num_carrying
        pacman_enemies_num = self.get_pacman_enemies_num(successor)
        food_carried_limit = len(food_list)
        if len(food_list) >= 4:  # if there's more than 4 foods, only carry 30%
            food_carried_limit = len(food_list) * 0.3

        if is_pacman_current and pacman_enemies_num < 2 and num_scared_ghosts != 2 and num_food_carried > food_carried_limit:
            min_distance_to_home = min(
                [self.get_maze_distance(my_pos, central_pos) for central_pos in self.central_home_positions])
            # features['distance_to_home'] = min_distance_to_home
            if nearest_ghost_dist and min_distance_to_home < 3:  # if ghost is chasing you, and you are close to home
                min_distance = min_distance_to_home * 0.8
                features['distance_to_home'] = min_distance
            if num_food_carried > food_carried_limit:  # if you are carrying > limit of food
                features['distance_to_home'] = min_distance_to_home

        # Compute distance to invaders (offensive agent should also chase invaders if they are close)
        if num_food_carried == 0 and not is_pacman_successor and len(invaders) > 0:
            closest_invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            if closest_invader_distance < 4:
                features['invader_distance'] = closest_invader_distance

        # Compute distance to scared ghosts
        scared_ghosts = self.get_scared_ghosts(successor)
        if len(scared_ghosts) > 0:
            closest_scared_ghost_distance = min([self.get_maze_distance(my_pos, g.get_position()) for g in scared_ghosts])
            if closest_scared_ghost_distance < 4:
                features['scared_ghost_distance'] = closest_scared_ghost_distance

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Compute whether it is beneficial to switch defence and offence roles
        self.switch_sides(game_state, is_pacman_current)

        return features

    def get_offence_weights(self, game_state, action):
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        nearest_ghost_dist = self.get_nearest_ghost_dist(successor, my_pos)
        nearest_capsule_dist = self.get_nearest_capsule_dist(game_state, my_pos)

        # if we are near a ghosts and near a capsule, prioritize going to capsule
        if nearest_ghost_dist and nearest_capsule_dist and nearest_capsule_dist <= nearest_ghost_dist:
            distance_to_capsules_weight = -200
        else:
            distance_to_capsules_weight = -5

        return {
            'successor_score': 100,  # maximising score
            'distance_to_food': -3,  # minimising distance to food
            'distance_to_ghost': 70,  # maximising distance to ghosts
            'distance_to_capsules': distance_to_capsules_weight,  # minimising distance to capsules
            'distance_to_home': -4,  # minimising distance to start position
            'go_to_capsule': 200,  # maximising value for going to capsule
            'invader_distance': -50,  # minimising distance to invader
            'scared_ghost_distance': -50,  # minimising distance to scared ghost
            'stop': -100,  # minimising chance of stopping
            'reverse': -4  # minimising chance of going back and forth
            }

    def get_defence_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        is_pacman_current = game_state.get_agent_state(self.index).is_pacman
        my_state = successor.get_agent_state(self.index)

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        features['num_invaders'] = len(invaders)

        # Computes distance to food -> protect the food
        food_list = self.get_food_you_are_defending(successor).as_list()
        food_list_at_start = self.get_food_you_are_defending(self.start_state).as_list()

        # Calculate whether it is beneficial to guard food that is left
        num_pacmans = self.get_pacman_enemies_num(successor)
        if num_pacmans >= 1 and len(food_list) / len(food_list_at_start) < 0.3 and len(food_list) > 2:
            distance_food_protect = random.choice([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food_protect'] = distance_food_protect  # minimise this distance

        # Computes distance to invaders we can see -> chase the closest invader
        if len(invaders) > 0:
            closest_invader_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in invaders])
            features['invader_distance'] = closest_invader_distance  # minimise this distance
            features['distance_to_food_protect'] = 0  # don't care about the food, just chase invaders

        if num_pacmans == 0 or len(food_list) <= 2:
            my_min_distance_center = min(
                [self.get_maze_distance(my_pos, central_pos) for central_pos in self.central_home_positions])
            features['distance_to_centre'] = my_min_distance_center  # minimise this distance

        if action == Directions.STOP: features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Compute whether it is beneficial to switch defence and offence roles
        self.switch_sides(game_state, is_pacman_current)

        return features

    def get_defence_weights(self):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'distance_to_centre': -10, 'distance_to_food_protect': -5, 'stop': -100, 'reverse': -2}

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.on_defence = False


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.on_defence = True

 # # Compute distance to your side (ie to your start position)
        # # -> we want to minimise this distance, if carrying food and if ghosts are not scared
        # num_scared_ghosts = self.get_scared_ghosts_num(successor)
        # num_food_carried = game_state.get_agent_state(self.index).num_carrying
        # if num_scared_ghosts != 2 and num_food_carried > len(food_list)*0.2:
        #     start_pos_distance = self.get_maze_distance(my_pos, self.start)
        #     features['distance_to_home'] = start_pos_distance