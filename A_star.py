### Document the structure of A* with Manhattan distance heuristic
import numpy as np
import heapq
from termcolor import colored # used just to make the output colourful

## Class to represent the state of the puzzle
class PuzzleState:
    def __init__(self, board, parent, move, depth, cost):
        self.board = board
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost

def print_board(board):
    for row in range(0, 9, 3):
        row_visual = "|"
        for tile in board[row:row +3]:
            if tile == 0:  # Blank tile
                row_visual += f" {colored(' ', 'cyan')} |"
            else:
                row_visual += f" {colored(str(tile), 'yellow')} |"
        print(row_visual)
        print("+---+---+---+")

### Define the goal state and Possible Moves
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]
moves = {
    'U': -3,
    'D': +3,
    'L': -1,
    'R': +1
}

def misplaced_tiles_heuristic(board):
    """Count the number of misplaced tiles."""
    heuristic_sum = 0
    for i in range(len(board)):
        if board[i] != 0 and board[i] != GOAL_STATE[i]:
            heuristic_sum += 1
    return heuristic_sum

def manhattan_dist_heuristic(board):
    dist = 0
    for i in range(len(board)):
        if board[i] != 0:
            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(board[i] - 1, 3)
            dist += abs(x1 - x2) + abs(y1 - y2)
    return dist

def move_tile(board, move, blank_pos):
    new_board = board[:]
    new_blank_pos = blank_pos + moves[move]
    new_board[blank_pos], new_board[new_blank_pos] = new_board[new_blank_pos], new_board[blank_pos]
    return new_board

def a_star(start_state):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, PuzzleState(start_state, None, None, 0, manhattan_dist_heuristic(start_state)))

    while open_list:
        current_state = heapq.heappop(open_list)
        if current_state.board == GOAL_STATE:
            return current_state
        closed_list.add(tuple(current_state.board))

        blank_pos = current_state.board.index(0)

        for move in moves:
            if move == 'U' and blank_pos < 3: continue
            if move == 'D' and blank_pos > 5: continue
            if move == 'L' and blank_pos % 3 == 0: continue
            if move == 'R' and blank_pos % 3 == 2: continue
            new_board = move_tile(current_state.board, move, blank_pos)
            if tuple(new_board) in closed_list:
                continue
            new_state = PuzzleState(new_board, current_state, move, current_state.depth + 1, current_state.depth + 1 + manhattan_dist_heuristic(new_board))
            heapq.heappush(open_list, new_state)

    return None


def a_star_misplaced_tile(start_state):
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, PuzzleState(start_state, None, None, 0, misplaced_tiles_heuristic(start_state)))

    while open_list:
        current_state = heapq.heappop(open_list)
        if current_state.board == GOAL_STATE:
            return current_state
        closed_list.add(tuple(current_state.board))

        blank_pos = current_state.board.index(0)

        for move in moves:
            if move == 'U' and blank_pos < 3: continue
            if move == 'D' and blank_pos > 5: continue
            if move == 'L' and blank_pos % 3 == 0: continue
            if move == 'R' and blank_pos % 3 == 2: continue
            new_board = move_tile(current_state.board, move, blank_pos)
            if tuple(new_board) in closed_list:
                continue
            new_state = PuzzleState(new_board, current_state, move, current_state.depth + 1, current_state.depth + 1 + misplaced_tiles_heuristic(new_board))
            heapq.heappush(open_list, new_state)

    return None



def print_solution(solution):
    path = []
    current = solution
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    for step in path:
        print(f"Move: {step.move}, Depth: {step.depth}")
        print_board(step.board)

if __name__ == "__main__":
    initial_state = [1, 2, 3, 4,7, 5 , 6, 0, 8]
    # solution = a_star(initial_state)
    solution = a_star_misplaced_tile(initial_state)

    if solution:
        print(colored("solution found", "green"))
        print_solution(solution)
    else:
        print(colored("No solution found", "red"))


