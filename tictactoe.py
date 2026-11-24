import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------
# STATE s = board (3x3 numpy int array)
#   board[r, c] values:
#       0  -> empty cell
#       1  -> X (Maximizer, wants UTILITY = +1)
#      -1  -> O (Minimizer, wants UTILITY = -1)
#   r = row 0..2 (top -> bottom)
#   c = col 0..2 (left -> right)
# ---------------------------------------------------------------


# S0: initial state - 3x3 board, all empty
def initial_state():
    return np.zeros((3, 3), dtype=int)


S0 = initial_state()


def PLAYER(board):
    """X moves first. After each X move, O moves. Count pieces to decide."""
    x_count = np.sum(board == 1)
    o_count = np.sum(board == -1)
    return 1 if x_count == o_count else -1


def ACTIONS(board):
    """All empty cells are legal moves."""
    return [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]


def RESULT(board, action):
    """Apply action; returns new board (original unchanged)."""
    r, c = action
    new_board = board.copy()
    new_board[r, c] = PLAYER(board)
    return new_board


def _winner(board):
    """Returns winning player (1 or -1) or None."""
    for p in (1, -1):
        for i in range(3):
            if all(board[i, :] == p) or all(board[:, i] == p):
                return p
        if board[0, 0] == board[1, 1] == board[2, 2] == p:
            return p
        if board[0, 2] == board[1, 1] == board[2, 0] == p:
            return p
    return None


def TERMINAL(board):
    """Game over if there's a winner OR no moves left."""
    return _winner(board) is not None or len(ACTIONS(board)) == 0


def UTILITY(board):
    """Only meaningful on terminal states."""
    w = _winner(board)
    return w if w is not None else 0


# ---------------------------------------------------------------
# MINIMAX(s):
#   if TERMINAL(s): return UTILITY(s)
#   if PLAYER(s) == X (MAX): wants highest value
#       return max over a in ACTIONS(s): MINIMAX(RESULT(s, a))
#   if PLAYER(s) == O (MIN): wants lowest value
#       return min over a in ACTIONS(s): MINIMAX(RESULT(s, a))
# ---------------------------------------------------------------
_cache = {}


def MINIMAX(board):
    key = board.tobytes()
    if key in _cache:
        return _cache[key]
    if TERMINAL(board):
        value = UTILITY(board)
    elif PLAYER(board) == 1:  # X = MAX
        value = max(MINIMAX(RESULT(board, a)) for a in ACTIONS(board))
    else:                     # O = MIN
        value = min(MINIMAX(RESULT(board, a)) for a in ACTIONS(board))
    _cache[key] = value
    return value


def BEST_MOVE(board):
    """Pick the action whose MINIMAX value is best for the player to move."""
    p = PLAYER(board)
    best_action, best_value = None, None
    for a in ACTIONS(board):
        v = MINIMAX(RESULT(board, a))
        if best_value is None or (p == 1 and v > best_value) or (p == -1 and v < best_value):
            best_action, best_value = a, v
    return best_action


# ---------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------
def draw_board(ax, board, title=""):
    ax.clear()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(1, 3):
        ax.axvline(i, color="#333333", lw=2)
        ax.axhline(i, color="#333333", lw=2)
    for r in range(3):
        for c in range(3):
            # board row 0 = top, but matplotlib y=0 = bottom
            cx, cy = c + 0.5, 2.5 - r
            if board[r, c] == 1:       # X - blue cross
                ax.plot([cx - 0.3, cx + 0.3], [cy - 0.3, cy + 0.3], color="#2563eb", lw=4)
                ax.plot([cx - 0.3, cx + 0.3], [cy + 0.3, cy - 0.3], color="#2563eb", lw=4)
            elif board[r, c] == -1:    # O - red circle
                ax.add_patch(Circle((cx, cy), 0.3, fill=False, edgecolor="#dc2626", lw=4))
    ax.set_title(title, fontsize=12)


# ---------------------------------------------------------------
# Game (click to play)
#   Keys:  m = switch mode (Human vs Human / Human vs AI)
#          r = restart
# ---------------------------------------------------------------
class Game:
    def __init__(self):
        self.board = initial_state()
        self.vs_ai = True  # human = X, AI = O
        self.fig, self.ax = plt.subplots(figsize=(5, 5.6))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def status(self):
        mode = "Human vs AI" if self.vs_ai else "Human vs Human"
        if TERMINAL(self.board):
            w = _winner(self.board)
            result = "Draw!" if w is None else ("X wins!" if w == 1 else "O wins!")
            return f"{result}  (press r to restart)  [{mode}]"
        turn = "X" if PLAYER(self.board) == 1 else "O"
        return f"{turn}'s turn  [{mode}]  (m = mode, r = restart)"

    def redraw(self):
        draw_board(self.ax, self.board, self.status())
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or TERMINAL(self.board):
            return
        if event.xdata is None or event.ydata is None:
            return
        if not (0 <= event.xdata < 3 and 0 <= event.ydata < 3):
            return
        c = int(event.xdata)
        r = 2 - int(event.ydata)  # flip y
        if self.board[r, c] != 0:
            return
        if self.vs_ai and PLAYER(self.board) != 1:
            return
        self.board = RESULT(self.board, (r, c))
        if self.vs_ai and not TERMINAL(self.board):
            self.board = RESULT(self.board, BEST_MOVE(self.board))
        self.redraw()

    def on_key(self, event):
        if event.key == "r":
            self.board = initial_state()
        elif event.key == "m":
            self.vs_ai = not self.vs_ai
            self.board = initial_state()
        else:
            return
        self.redraw()


if __name__ == "__main__":
    print("Minimax value of empty board (0 = draw with perfect play):", MINIMAX(S0))
    game = Game()
    plt.show()