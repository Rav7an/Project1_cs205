import heapq
from termcolor import colored

# Defined a class to represent the state of the puzzle
# We will be using different instances of this class to represent the states at various depths
class PuzzleState:
    def __init__(self, board, parent, move, depth, cost):
        self.board  = board
        self.parent = parent
        self.move   = move
        self.depth  = depth
        self.cost   = cost

    def __lt__(self, other):
        return self.cost < other.cost

### Define puzzle size and generate goal state & moves dynamically
N = 3  # Change this to solve an N-puzzle (e.g., 4 for 15-puzzle)
SIZE = N * N
"""
 Below line defines the actions for any N-puzzle
 Default N = 3
 Thus Goal State = [1, 2, 3, 4, 5, 6, 7, 8, 0]
"""
GOAL_STATE = list(range(1, SIZE)) + [0]
moves = {
    'U': -N,
    'D': +N,
    'L': -1,
    'R': +1
}

def print_board_state(board):
    """Print the board state in a formatted way."""
    for row in range(0, SIZE, N):
        row_visual = "|"
        for tile in board[row:row + N]:
            if tile == 0:
                row_visual += f" {colored(' ', 'cyan')} |"
            else:
                row_visual += f" {colored(str(tile), 'yellow')} |"
        print(row_visual)
        print("+" + "---+" * N)

def misplaced_tiles_heuristic(board):
    """Counts the number of misplaced tiles.
       Input: State of the Board
       Returns: The sum of the number of misplaced tiles
    """
    dist = 0
    for i in range(len(board)):
        if board[i] != 0:
            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(board[i] - 1, 3)
            dist += abs(x1 - x2) + abs(y1 - y2)
    return dist


def manhattan_dist_heuristic(board):
    """Calculates the Manhattan distance heuristic.
       Input: State of the Board
       Returns: The sum of the Manhattan distances of all tiles from their goal positions
    """
    dist = 0
    for i, v in enumerate(board):
        if v == 0:
            continue
        r1, c1 = divmod(i, N)
        goal_idx = GOAL_STATE.index(v)
        r2, c2 = divmod(goal_idx, N)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist

def move_tile(board, move, blank_pos):
    """Moves the blank tile in the specified direction and returns the new board state.
       Input: Current Board State, Move Direction, Position of Blank Tile
       Returns: New Board State after the move
    """
    new_board = board[:]
    new_blank = blank_pos + moves[move]
    new_board[blank_pos], new_board[new_blank] = new_board[new_blank], new_board[blank_pos]
    return new_board

# ———————— General Search ————————
def general_search(start_state, queue_fn):
    open_list, closed_list = [], set()
    heapq.heappush(open_list, PuzzleState(start_state, None, None, 0, 0))

    while open_list:
        current = heapq.heappop(open_list)
        if current.board == GOAL_STATE:
            return current
        closed_list.add(tuple(current.board))

        blank_pos = current.board.index(0)
        children = []
        row, col = divmod(blank_pos, N)
        for mv, offset in moves.items():
            nr = row + (-1 if mv == 'U' else 1 if mv == 'D' else 0)
            nc = col + (-1 if mv == 'L' else 1 if mv == 'R' else 0)

            # Check if the state to be generated is valid
            if not (0 <= nr < N and 0 <= nc < N):
                continue
            new_board = move_tile(current.board, mv, blank_pos)

            # Check to avoid queueing the same state again
            if tuple(new_board) in closed_list:
                continue
            new_depth = current.depth + 1
            child = PuzzleState(new_board, current, mv, new_depth, 0)
            children.append(child)

        queue_fn(open_list, children)
    return None


def ucs_queue(open_list, children):
    """Uniform Cost Search: Adds children to the open list with their depth as cost."""
    for child in children:
        child.cost = child.depth
        heapq.heappush(open_list, child)

def astar_misplaced_queue(open_list, children):
    """A* Search with Misplaced Tiles Heuristic:
       Adds children to the open list with their depth + heuristic as cost."""
    for child in children:
        h = misplaced_tiles_heuristic(child.board)
        child.cost = child.depth + h
        heapq.heappush(open_list, child)

def astar_manhattan_queue(open_list, children):
    """A* Search with Manhattan Distance Heuristic:
       Adds children to the open list with their depth + heuristic as cost.
         Input: Open List, Children Nodes
    """
    for child in children:
        h = manhattan_dist_heuristic(child.board)
        child.cost = child.depth + h
        heapq.heappush(open_list, child)

# ———————— Helpers & Main ————————
def print_solution(solution):
    """Prints the solution path from the initial state to the goal state.
        Input: Solution Node State
        Creates a list and adds all the states from the solution state till the initial state
        Prints the path in reverse order
    """
    path, curr = [], solution
    while curr:
        path.append(curr)
        curr = curr.parent
    path.reverse()
    for step in path:
        print(f"Move: {step.move}, Depth: {step.depth}")
        print_board_state(step.board)

if __name__ == "__main__":
    # Example initial state
    initial_state = [1, 2, 3, 4, 7, 5, 6, 0, 8]
    # The below code generates the solution if present for all the 3 algorithms
    sol = general_search(initial_state, ucs_queue)
    if sol:
        print(colored("UCS solution found", "green"))
        print_solution(sol)
    else:
        print(colored("UCS: No solution found", "red"))

    sol = general_search(initial_state, astar_misplaced_queue)
    if sol:
        print(colored("A* Misplaced solution found", "green"))
        print_solution(sol)
    else:
        print(colored("A* Misplaced: No solution found", "red"))

    sol = general_search(initial_state, astar_manhattan_queue)
    if sol:
        print(colored("A* Manhattan solution found", "green"))
        print_solution(sol)
    else:
        print(colored("A* Manhattan: No solution found", "red"))


        print(colored("A* Manhattan: No solution found", "red"))