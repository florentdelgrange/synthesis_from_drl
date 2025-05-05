import math

from reinforcement_learning.environments import Directions

debug = False

actions = {
    'rotate_left': 2,
    'rotate_right': 3,
    'move_forward': 4,
    'move_backward': 5,
}

def calculate_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    angle_normalized = (angle_deg + 360) % 360
    return angle_normalized

def determine_turn(agent_x, agent_y, angle, target_x, target_y, delta):
    target_angle = calculate_angle(agent_x, agent_y, target_x, target_y)
    angle_difference = (target_angle - angle + 360) % 360

    if angle_difference < delta or angle_difference > 360 - delta:
        return actions['move_forward']
    elif angle_difference > 180:
        return actions['rotate_right']
    else:
        return actions['rotate_left']



def prescribe_action(x, y, angle, delta, out_room, in_room, low_level_goals):
    """
    Prescribe the action to take based on the current position and the rooms the agent is moving between.
    Returns: int, the action to take.
    """
    if debug:
        print("x", x, "y", y, "angle", angle, "delta", delta, "out_room", out_room, "in_room", in_room)
    if out_room == 0 and in_room == 1:
        return determine_turn(x, y, angle, low_level_goals[1][Directions.LEFT][0], low_level_goals[1][Directions.LEFT][1], delta)
    elif out_room == 0 and in_room == 3:
        return determine_turn(x, y, angle, low_level_goals[3][Directions.UP][0], low_level_goals[3][Directions.UP][1], delta)
    elif out_room == 1 and in_room == 0:
        return determine_turn(x, y, angle, low_level_goals[0][Directions.RIGHT][0], low_level_goals[0][Directions.RIGHT][1], delta)
    elif out_room == 1 and in_room == 2:
        if x < 1472:
            return determine_turn(x, y, angle, 1472, -64, delta)
        else:
            return determine_turn(x, y, angle, low_level_goals[2][Directions.UP][0], low_level_goals[2][Directions.UP][1], delta)
    elif out_room == 1 and in_room == 3:
        return determine_turn(x, y, angle, 960, -576, delta)
    elif out_room == 2 and in_room == 1:
        if y < -64:
            return determine_turn(x, y, angle, 1472, -64, delta)
        else:
            return determine_turn(x, y, angle, low_level_goals[1][Directions.RIGHT][0], low_level_goals[1][Directions.RIGHT][1], delta)
    elif out_room == 2 and in_room == 4:
        if y > - 1344:
            return determine_turn(x, y, angle, 1664, -1344, delta)
        else:
            return determine_turn(x, y, angle, low_level_goals[4][Directions.RIGHT][0], low_level_goals[4][Directions.RIGHT][1], delta)
    elif out_room == 2 and in_room == 5:
        return determine_turn(x, y, angle, low_level_goals[5][Directions.RIGHT][0], low_level_goals[5][Directions.RIGHT][1], delta)
    elif out_room == 3 and in_room == 5:
        return determine_turn(x, y, angle, low_level_goals[5][Directions.UP][0], low_level_goals[5][Directions.UP][1], delta)
    elif out_room == 3 and in_room == 0:
        return determine_turn(x, y, angle, low_level_goals[0][Directions.DOWN][0], low_level_goals[0][Directions.DOWN][1], delta)
    elif out_room == 4 and in_room == 2:
        return determine_turn(x, y, angle, low_level_goals[2][Directions.DOWN][0], low_level_goals[2][Directions.DOWN][1], delta)
    elif out_room == 4 and in_room == 5:
        if x > 768:
            return determine_turn(x, y, angle, 768, -1376, delta)
        else:
            return determine_turn(x, y, angle, low_level_goals[5][Directions.DOWN][0], low_level_goals[5][Directions.DOWN][1], delta)
    elif out_room == 5 and in_room == 2:
        return determine_turn(x, y, angle, low_level_goals[2][Directions.LEFT][0], low_level_goals[2][Directions.LEFT][1], delta)
    elif out_room == 5 and in_room == 3:
        return determine_turn(x, y, angle, low_level_goals[3][Directions.DOWN][0], low_level_goals[3][Directions.DOWN][1], delta)
    elif out_room == 5 and in_room == 4:
        if x < 800:
            return determine_turn(x, y, angle, 800, -1376, delta)
        else:
            return determine_turn(x, y, angle, low_level_goals[4][Directions.UP][0], low_level_goals[4][Directions.UP][1], delta)
    elif out_room == 5 and in_room == 6:
        return determine_turn(x, y, angle, low_level_goals[6][Directions.RIGHT][0], low_level_goals[6][Directions.RIGHT][1], delta)
    elif out_room == 6 and in_room == 5:
        return determine_turn(x, y, angle, low_level_goals[5][Directions.LEFT][0], low_level_goals[5][Directions.LEFT][1], delta)
    elif out_room == 6 and in_room == 7:
        return determine_turn(x, y, angle, low_level_goals[7][Directions.RIGHT][0], low_level_goals[7][Directions.RIGHT][1], delta)
    elif out_room == 7 and in_room == 6:
        return determine_turn(x, y, angle, low_level_goals[6][Directions.LEFT][0], low_level_goals[6][Directions.LEFT][1], delta)

def _move_right(angle, delta):
    if angle < delta or angle > 360 - delta:
        return actions['move_forward']
    elif angle > 180:
        return actions['rotate_left']
    else:
        return actions['rotate_right']


def _move_down(angle, delta):
    if 270 - delta < angle < 270 + delta:
        return actions['move_forward']
    elif angle > 270 or angle < 90:
        return actions['rotate_right']
    else:
        return actions['rotate_left']


def _move_up(angle, delta):
    if 90 - delta < angle < 90 + delta:
        return actions['move_forward']
    elif angle < 90:
        return actions['rotate_left']
    else:
        return actions['rotate_right']


def _move_left(angle, delta):
    if 180 - delta < angle < 180 + delta:
        return actions['move_forward']
    elif 90 < angle < 180:
        return actions['rotate_left']
    else:
        return actions['rotate_right']


