import random
import copy
import time
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


WIN_COUNT = 4

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows: int) -> int:
    return rows - 3

def is_opponent_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

def is_own_score_cell(x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> bool:
    if player == "circle":
        return (y == top_score_row()) and (x in score_cols)
    else:
        return (y == bottom_score_row(rows)) and (x in score_cols)

def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"


def count_scoring_pieces(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    total = 0
    for y, row in enumerate(board):
        for x, piece in enumerate(row):
            if piece and piece.owner == player and piece.side == "stone" and is_own_score_cell(x, y, player, rows, cols, score_cols):
                total += 1
    return total


def get_river_flow_destinations(
    board: List[List[Any]],
    rx: int,
    ry: int,
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
    river_push: bool = False,
) -> List[Tuple[int, int]]:
    destinations: List[Tuple[int, int]] = []
    visited: set[Tuple[int, int]] = set()
    queue: List[Tuple[int, int]] = [(rx, ry)]

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]
        if river_push and x == rx and y == ry:
            cell = board[sy][sx]

        if cell is None:
            if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.append((x, y))
            continue

        if cell.side != "river":
            continue

        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue
                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue
                if next_cell.side == "river":
                    queue.append((nx, ny))
                    break
                break

    unique: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for dest in destinations:
        if dest not in seen:
            seen.add(dest)
            unique.append(dest)
    return unique


def compute_valid_targets(
    board: List[List[Any]],
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Dict[str, Any]:
    if not in_bounds(sx, sy, rows, cols):
        return {"moves": set(), "pushes": []}

    piece = board[sy][sx]
    if piece is None or piece.owner != player:
        return {"moves": set(), "pushes": []}

    moves: set[Tuple[int, int]] = set()
    pushes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in dirs:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target = board[ty][tx]
        if target is None:
            moves.add((tx, ty))
        elif target.side == "river":
            flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for d in flow:
                moves.add(d)
        else:
            if piece.side == "stone":
                px, py = tx + dx, ty + dy
                if (
                    in_bounds(px, py, rows, cols)
                    and board[py][px] is None
                    and not is_opponent_score_cell(px, py, piece.owner, rows, cols, score_cols)
                ):
                    pushes.append(((tx, ty), (px, py)))
            else:
                pushed_player = target.owner
                flow = get_river_flow_destinations(
                    board,
                    tx,
                    ty,
                    sx,
                    sy,
                    pushed_player,
                    rows,
                    cols,
                    score_cols,
                    river_push=True,
                )
                for d in flow:
                    if not is_opponent_score_cell(d[0], d[1], pushed_player, rows, cols, score_cols):
                        pushes.append(((tx, ty), (d[0], d[1])))

    return {"moves": moves, "pushes": pushes}


def validate_and_apply_move(
    board: List[List[Any]],
    move: Dict[str, Any],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Tuple[bool, str]:
    if not isinstance(move, dict):
        return False, "move must be dict"

    action = move.get("action")

    if action == "move":
        fr = move.get("from")
        to = move.get("to")
        if not fr or not to:
            return False, "move needs from & to"
        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        if not in_bounds(fx, fy, rows, cols) or not in_bounds(tx, ty, rows, cols):
            return False, "oob"
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            return False, "can't go into opponent score"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if board[ty][tx] is None:
            board[ty][tx] = piece
            board[fy][fx] = None
            return True, "moved"

        pushed = move.get("pushed_to")
        if not pushed:
            return False, "destination occupied; pushed_to required"
        ptx, pty = int(pushed[0]), int(pushed[1])
        dx = tx - fx
        dy = ty - fy
        if (ptx, pty) != (tx + dx, ty + dy):
            return False, "invalid pushed_to"
        if not in_bounds(ptx, pty, rows, cols):
            return False, "oob"
        if is_opponent_score_cell(ptx, pty, player, rows, cols, score_cols):
            return False, "can't push into opponent score"
        if board[pty][ptx] is not None:
            return False, "pushed_to not empty"

        board[pty][ptx] = board[ty][tx]
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, "move+push applied"

    if action == "push":
        fr = move.get("from")
        to = move.get("to")
        pushed = move.get("pushed_to")
        if not fr or not to or not pushed:
            return False, "push needs from,to,pushed_to"

        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])

        if not (
            in_bounds(fx, fy, rows, cols)
            and in_bounds(tx, ty, rows, cols)
            and in_bounds(px, py, rows, cols)
        ):
            return False, "oob"

        target_piece = board[ty][tx]
        pushed_player = target_piece.owner if target_piece else None
        if (
            is_opponent_score_cell(tx, ty, player, rows, cols, score_cols)
            or is_opponent_score_cell(px, py, pushed_player, rows, cols, score_cols)
        ):
            return False, "push would enter opponent score cell"

        mover = board[fy][fx]
        if mover is None or mover.owner != player:
            return False, "invalid piece"

        if target_piece is None:
            return False, "to must be occupied"
        if board[py][px] is not None:
            return False, "pushed_to not empty"

        if mover.side == "river" and target_piece.side == "river":
            return False, "rivers cannot push rivers"

        info = compute_valid_targets(board, fx, fy, player, rows, cols, score_cols)
        if ((tx, ty), (px, py)) not in info["pushes"]:
            return False, "push pair invalid"

        board[py][px] = target_piece
        board[ty][tx] = mover
        board[fy][fx] = None

        mover = board[ty][tx]
        if mover.side == "river":
            mover.side = "stone"
            mover.orientation = None

        return True, "push applied"

    if action == "flip":
        fr = move.get("from")
        if not fr:
            return False, "flip needs from"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal", "vertical"):
                return False, "stone->river needs orientation"
            piece.side = "river"
            piece.orientation = ori
            flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
            for dx, dy in flow:
                if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                    piece.side = "stone"
                    piece.orientation = None
                    return False, "flip would allow flow into opponent score"
            return True, "flipped to river"
        piece.side = "stone"
        piece.orientation = None
        return True, "flipped to stone"

    if action == "rotate":
        fr = move.get("from")
        if not fr:
            return False, "rotate needs from"
        fx, fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx, fy, rows, cols):
            return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid"
        if piece.side != "river":
            return False, "rotate only on river"
        piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
        flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        for dx, dy in flow:
            if is_opponent_score_cell(dx, dy, player, rows, cols, score_cols):
                piece.orientation = "horizontal" if piece.orientation == "vertical" else "vertical"
                return False, "rotation allows flow into opponent score"
        return True, "rotated"

    return False, "unknown action"


def count_reachable_in_one(
    board: List[List[Any]],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> int:
    m = 0
    for y, row in enumerate(board):
        for x, piece in enumerate(row):
            if piece and piece.owner == player and piece.side == "stone":
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    continue
                info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)
                for (tx, ty) in info.get("moves", set()):
                    if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                        m += 1
                        break
                else:
                    for _, pushed in info.get("pushes", []):
                        ptx, pty = pushed
                        if is_own_score_cell(ptx, pty, player, rows, cols, score_cols):
                            m += 1
                            break
            if piece and piece.owner == player and piece.side == "river":
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    m += 1
    return m


def check_win(
    board: List[List[Any]],
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Optional[str]:
    top = top_score_row()
    bottom = bottom_score_row(rows)
    circle_count = 0
    square_count = 0

    for x in score_cols:
        if in_bounds(x, top, rows, cols):
            piece = board[top][x]
            if piece and piece.owner == "circle" and piece.side == "stone":
                circle_count += 1
        if in_bounds(x, bottom, rows, cols):
            piece = board[bottom][x]
            if piece and piece.owner == "square" and piece.side == "stone":
                square_count += 1

    if circle_count >= WIN_COUNT:
        return "circle"
    if square_count >= WIN_COUNT:
        return "square"
    return None

def get_valid_moves_for_piece(board, x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    moves = []
    piece = board[y][x]
    
    if piece is None or piece.owner != player:
        return moves
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    if piece.side == "stone":
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            
            if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                continue
            
            if board[ny][nx] is None:
                moves.append({"action": "move", "from": [x, y], "to": [nx, ny]})
            elif board[ny][nx].owner != player:
                px, py = nx + dx, ny + dy
                if (in_bounds(px, py, rows, cols) and 
                    board[py][px] is None and 
                    not is_opponent_score_cell(px, py, player, rows, cols, score_cols)):
                    moves.append({"action": "push", "from": [x, y], "to": [nx, ny], "pushed_to": [px, py]})
        
        for orientation in ["horizontal", "vertical"]:
            moves.append({"action": "flip", "from": [x, y], "orientation": orientation})
    
    else:
        moves.append({"action": "flip", "from": [x, y]})
        moves.append({"action": "rotate", "from": [x, y]})
    
    return moves

def generate_all_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    all_moves = []
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player:
                piece_moves = get_valid_moves_for_piece(board, x, y, player, rows, cols, score_cols)
                all_moves.extend(piece_moves)
    
    return all_moves

def engine_legal_moves(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> List[Dict[str, Any]]:
    moves: List[Dict[str, Any]] = []
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p or p.owner != player:
                continue
            info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)
            from_score = is_own_score_cell(x, y, player, rows, cols, score_cols)
            for (tx, ty) in info.get('moves', set()):
                m = {"action": "move", "from": [x, y], "to": [tx, ty]}
                if abs(tx - x) + abs(ty - y) > 1:
                    m["via_river"] = True
                if from_score:
                    m["from_score"] = True
                if is_own_score_cell(tx, ty, player, rows, cols, score_cols):
                    m["to_score"] = True
                if from_score and not m.get("to_score"):
                    m["exit_score"] = True
                moves.append(m)
            for (of, pf) in info.get('pushes', []):
                m = {"action": "push", "from": [x, y], "to": [of[0], of[1]], "pushed_to": [pf[0], pf[1]]}
                if from_score:
                    m["from_score"] = True
                    if not is_own_score_cell(of[0], of[1], player, rows, cols, score_cols):
                        m["exit_score"] = True
                if is_own_score_cell(of[0], of[1], player, rows, cols, score_cols):
                    m["to_score"] = True
                moves.append(m)
            if p.side == "stone":
                for ori in ("horizontal", "vertical"):
                    old_side, old_ori = p.side, p.orientation
                    p.side, p.orientation = "river", ori
                    flow = get_river_flow_destinations(board, x, y, x, y, player, rows, cols, score_cols)
                    p.side, p.orientation = old_side, old_ori
                    if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                        m = {"action": "flip", "from": [x, y], "orientation": ori}
                        if from_score:
                            m["from_score"] = True
                        moves.append(m)
            else:
                m_flip = {"action": "flip", "from": [x, y]}
                if from_score:
                    m_flip["from_score"] = True
                moves.append(m_flip)
                new_ori = "vertical" if p.orientation == "horizontal" else "horizontal"
                old_ori = p.orientation
                p.orientation = new_ori
                flow = get_river_flow_destinations(board, x, y, x, y, player, rows, cols, score_cols)
                p.orientation = old_ori
                if not any(is_opponent_score_cell(dx, dy, player, rows, cols, score_cols) for dx, dy in flow):
                    m_rot = {"action": "rotate", "from": [x, y]}
                    if from_score:
                        m_rot["from_score"] = True
                    moves.append(m_rot)
    return moves

def count_stones_in_scoring_area(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> int:
    count = 0
    
    if player == "circle":
        score_row = top_score_row()
    else:
        score_row = bottom_score_row(rows)
    
    for x in score_cols:
        if in_bounds(x, score_row, rows, cols):
            piece = board[score_row][x]
            if piece and piece.owner == player and piece.side == "stone":
                count += 1
    
    return count

def basic_evaluate_board(board: List[List[Any]], player: str, rows: int, cols: int, score_cols: List[int]) -> float:
    score = 0.0
    opponent = get_opponent(player)
    
    player_scoring_stones = count_stones_in_scoring_area(board, player, rows, cols, score_cols)
    opponent_scoring_stones = count_stones_in_scoring_area(board, opponent, rows, cols, score_cols)
    
    score += player_scoring_stones * 100  
    score -= opponent_scoring_stones * 100  
    
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece and piece.owner == player and piece.side == "stone":
                if player == "circle":
                    score += (rows - y) * 0.1
                else:
                    score += y * 0.1
    
    return score

def simulate_move(board: List[List[Any]], move: Dict[str, Any], player: str, rows: int, cols: int, score_cols: List[int]) -> Tuple[bool, Any]:
    board_copy = copy.deepcopy(board)
    try:
        success, message = validate_and_apply_move(board_copy, move, player, rows, cols, score_cols)
    except Exception as exc:
        return False, str(exc)
    return success, board_copy if success else message

class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)
    
    @abstractmethod
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        pass

class StudentAgent(BaseAgent):
    def __init__(self, player: str):
        super().__init__(player)
        self.turn = 0
        self._tt = {}
        self._eval_cache = {}
        self._pv = {}
        self._move_cache = {}
        self._history_queue = deque(maxlen=32)
        self._history_counts: Dict[str, int] = {}
        self._repeat_limit = 3
        self._repeat_penalty = 5000
        self._engine_apply = validate_and_apply_move
        self._count_scoring_fn = count_scoring_pieces
        self._count_reachable_fn = count_reachable_in_one
        self._check_win_fn = check_win
        self._max_iter_depth = 6
        self._time_margin = 0.02
    
    def choose(self, board: List[List[Any]], rows: int, cols: int, score_cols: List[int], current_player_time: float, opponent_time: float) -> Optional[Dict[str, Any]]:
        self.turn += 1

        current_hash = self._hash_board(board)
        self._record_history(current_hash)

        self._eval_cache.clear()
        self._move_cache.clear()

        moves = self._legal_moves(board, self.player, rows, cols, score_cols)
        if not moves:
            return None

        current_score = self._count_scoring(board, self.player, rows, cols, score_cols)
        immediate_move, immediate_board = self._find_immediate_scoring_move(board, rows, cols, score_cols, moves, current_score)
        if immediate_move is not None:
            if immediate_board is not None:
                self._record_history(self._hash_board(immediate_board))
            return immediate_move

        non_exit_moves = [m for m in moves if not m.get("exit_score")]

        start = time.time()
        remaining = max(current_player_time, 0.01)
        divisor = 48 if self.turn < 10 else 70
        per_turn_budget = min(max(remaining / divisor, 0.04), 0.35)

        best_move = None
        best_val = -1e18
        evaluator = self._make_evaluator(rows, cols, score_cols)
        path_counts = defaultdict(int)
        path_counts[current_hash] = 1

        try:
            for depth in range(1, self._max_iter_depth + 1):
                move, val = self._alphabeta_root(board, rows, cols, score_cols, depth, evaluator, start, per_turn_budget, path_counts)
                if move is not None:
                    best_move, best_val = move, val
                if time.time() - start > per_turn_budget - self._time_margin:
                    break
        except TimeoutError:
            pass

        if best_move is None:
            candidate_pool = non_exit_moves if non_exit_moves else moves
            ordered = self._order_moves(list(candidate_pool), board, rows, cols, score_cols)
            return ordered[0] if ordered else None
        if best_move.get("exit_score") and non_exit_moves:
            conservative = self._order_moves(list(non_exit_moves), board, rows, cols, score_cols)
            if conservative:
                best_move = conservative[0]
        ok, next_board = self._apply_fast(board, best_move, self.player, rows, cols, score_cols)
        if ok and next_board:
            next_hash = self._hash_board(next_board)
            self._record_history(next_hash)
        return best_move

    def _hash_board(self, board) -> str:
        acc = []
        for row in board:
            for cell in row:
                if not cell:
                    acc.append('_.')
                else:
                    acc.append(f"{cell.owner[0]}{cell.side[0]}{'h' if (cell.orientation or 'h')=='horizontal' else 'v'}")
        return ''.join(acc)

    def _legal_moves(self, board, player, rows, cols, score_cols):
        key = (player, self._hash_board(board))
        moves = self._move_cache.get(key)
        if moves is None:
            moves = engine_legal_moves(board, player, rows, cols, score_cols)
            self._move_cache[key] = moves
        return list(moves)

    def _record_history(self, board_hash: str) -> None:
        if not board_hash:
            return
        if len(self._history_queue) == self._history_queue.maxlen:
            old = self._history_queue.popleft()
            if old in self._history_counts:
                self._history_counts[old] -= 1
                if self._history_counts[old] <= 0:
                    self._history_counts.pop(old, None)
        self._history_queue.append(board_hash)
        self._history_counts[board_hash] = self._history_counts.get(board_hash, 0) + 1

    def _repeat_value(self, maximizing: bool) -> float:
        return -self._repeat_penalty if maximizing else self._repeat_penalty

    def _root_branch_cap(self, depth: int) -> int:
        if depth <= 2:
            return 42
        if depth <= 3:
            return 32
        if depth <= 4:
            return 26
        return 20

    def _branch_cap(self, depth: int, maximizing: bool) -> int:
        if depth >= 4:
            return 22 if maximizing else 20
        if depth >= 3:
            return 20 if maximizing else 18
        return 18 if maximizing else 16

    def _count_scoring(self, board, player, rows, cols, score_cols) -> int:
        if self._count_scoring_fn:
            try:
                return int(self._count_scoring_fn(board, player, rows, cols, score_cols))
            except Exception:
                pass
        return count_stones_in_scoring_area(board, player, rows, cols, score_cols)

    def _check_win(self, board, rows, cols, score_cols) -> Optional[str]:
        fn = self._check_win_fn
        if fn is None:
            return None
        try:
            return fn(board, rows, cols, score_cols)
        except Exception:
            return None

    def _find_immediate_scoring_move(self, board, rows, cols, score_cols, moves, current_score):
        best_move = None
        best_board = None
        best_gain = 0
        for m in moves:
            if not m.get("to_score"):
                continue
            ok, nb = self._apply_fast(board, m, self.player, rows, cols, score_cols)
            if not ok or nb is None:
                continue
            winner = self._check_win(nb, rows, cols, score_cols)
            if isinstance(winner, str) and winner == self.player:
                return m, nb
            new_score = self._count_scoring(nb, self.player, rows, cols, score_cols)
            gain = new_score - current_score
            if gain > 0 and gain >= best_gain:
                best_gain = gain
                best_move = m
                best_board = nb
        return best_move, best_board

    def _alphabeta_root(self, board, rows, cols, score_cols, depth, evaluator, start, budget, path_counts):
        key = self._hash_board(board)
        base_moves = self._legal_moves(board, self.player, rows, cols, score_cols)
        if key in self._pv:
            try:
                pv_move = self._pv[key]
                base_moves = [pv_move] + [m for m in base_moves if m is not pv_move]
            except Exception:
                pass
        moves = self._order_moves(base_moves, board, rows, cols, score_cols)
        alpha, beta = -1e18, 1e18
        best_move = None
        best_val = -1e18
        branch_cap = self._root_branch_cap(depth)
        for idx, m in enumerate(moves):
            if idx >= branch_cap:
                break
            if time.time() - start > budget - self._time_margin:
                raise TimeoutError
            ok, nb = self._apply_fast(board, m, self.player, rows, cols, score_cols)
            if not ok:
                continue
            nb_hash = self._hash_board(nb)
            historic_repeats = self._history_counts.get(nb_hash, 0)
            path_counts[nb_hash] += 1
            try:
                if path_counts[nb_hash] >= self._repeat_limit:
                    val = self._repeat_value(maximizing=False)
                else:
                    val = self._alphabeta(nb, rows, cols, score_cols, depth - 1, alpha, beta, False, evaluator, start, budget, path_counts)
            finally:
                path_counts[nb_hash] -= 1
                if path_counts[nb_hash] <= 0:
                    path_counts.pop(nb_hash, None)
            if historic_repeats >= self._repeat_limit - 1:
                val -= self._repeat_penalty * 0.75
            if val > best_val:
                best_val, best_move = val, m
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        if moves:
            self._pv[key] = best_move if best_move is not None else moves[0]
        return best_move, best_val

    def _alphabeta(self, board, rows, cols, score_cols, depth, alpha, beta, maximizing, evaluator, start, budget, path_counts):
        if time.time() - start > budget - self._time_margin:
            raise TimeoutError
        board_hash = self._hash_board(board)
        if path_counts.get(board_hash, 0) >= self._repeat_limit:
            return self._repeat_value(maximizing)
        if depth <= 0:
            key = self._hash_board(board)
            if key in self._eval_cache:
                return self._eval_cache[key]
            val = evaluator(board)
            self._eval_cache[key] = val
            return val
        player = self.player if maximizing else self.opponent
        moves = self._legal_moves(board, player, rows, cols, score_cols)
        if not moves:
            key = self._hash_board(board)
            if key in self._eval_cache:
                return self._eval_cache[key]
            val = evaluator(board)
            self._eval_cache[key] = val
            return val
        moves = self._order_moves(moves, board, rows, cols, score_cols, player)
        if maximizing:
            val = -1e18
            branch_cap = self._branch_cap(depth, True)
            for idx, m in enumerate(moves):
                if idx >= branch_cap:
                    break
                ok, nb = self._apply_fast(board, m, player, rows, cols, score_cols)
                if not ok:
                    continue
                nb_hash = self._hash_board(nb)
                historic_repeats = self._history_counts.get(nb_hash, 0)
                path_counts[nb_hash] = path_counts.get(nb_hash, 0) + 1
                try:
                    if path_counts[nb_hash] >= self._repeat_limit:
                        child_val = self._repeat_value(False)
                    else:
                        child_val = self._alphabeta(nb, rows, cols, score_cols, depth - 1, alpha, beta, False, evaluator, start, budget, path_counts)
                finally:
                    path_counts[nb_hash] -= 1
                    if path_counts[nb_hash] <= 0:
                        path_counts.pop(nb_hash, None)
                if historic_repeats >= self._repeat_limit - 1:
                    child_val -= self._repeat_penalty * 0.75
                if child_val > val:
                    val = child_val
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = 1e18
            branch_cap = self._branch_cap(depth, False)
            for idx, m in enumerate(moves):
                if idx >= branch_cap:
                    break
                ok, nb = self._apply_fast(board, m, player, rows, cols, score_cols)
                if not ok:
                    continue
                nb_hash = self._hash_board(nb)
                historic_repeats = self._history_counts.get(nb_hash, 0)
                path_counts[nb_hash] = path_counts.get(nb_hash, 0) + 1
                try:
                    if path_counts[nb_hash] >= self._repeat_limit:
                        child_val = self._repeat_value(True)
                    else:
                        child_val = self._alphabeta(nb, rows, cols, score_cols, depth - 1, alpha, beta, True, evaluator, start, budget, path_counts)
                finally:
                    path_counts[nb_hash] -= 1
                    if path_counts[nb_hash] <= 0:
                        path_counts.pop(nb_hash, None)
                if historic_repeats >= self._repeat_limit - 1:
                    child_val += self._repeat_penalty * 0.75
                if child_val < val:
                    val = child_val
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def _apply(self, board, move, player, rows, cols, score_cols):
        engine_apply = self._engine_apply
        if engine_apply is None:
            return True, copy.deepcopy(board)
        nb = self._copy_board(board)
        try:
            ok, _ = engine_apply(nb, move, player, rows, cols, score_cols)
        except Exception:
            return True, copy.deepcopy(board)
        return ok, nb

    def _apply_fast(self, board, move, player, rows, cols, score_cols):
        """Delegate move application to engine validator for correctness across all actions."""
        return self._apply(board, move, player, rows, cols, score_cols)

    def _copy_board(self, board):
        return [[cell.copy() if cell else None for cell in row] for row in board]

    def _make_evaluator(self, rows, cols, score_cols):
        opp = self.opponent
        me = self.player
        compute_targets = compute_valid_targets
        get_flow = get_river_flow_destinations
        count_scoring = self._count_scoring_fn
        count_reachable = self._count_reachable_fn

        def own_score_row(p):
            return top_score_row() if p == "circle" else bottom_score_row(rows)

        def nm_values(board, p):
            if count_scoring and count_reachable:
                try:
                    n = count_scoring(board, p, rows, cols, score_cols)
                    m = count_reachable(board, p, rows, cols, score_cols)
                    return float(n), float(m)
                except Exception:
                    pass
            srow = own_score_row(p)
            n = 0
            for x in score_cols:
                cell = board[srow][x]
                if cell and cell.owner == p and cell.side == "stone":
                    n += 1
            m = 0
            if compute_targets:
                for y in range(rows):
                    for x in range(cols):
                        cell = board[y][x]
                        if cell and cell.owner == p and cell.side == "stone":
                            info = compute_targets(board, x, y, p, rows, cols, score_cols)
                            for (tx, ty) in info.get('moves', set()):
                                if is_own_score_cell(tx, ty, p, rows, cols, score_cols):
                                    m += 1
                                    break
                            else:
                                for (_of, pushed) in info.get('pushes', []):
                                    if is_own_score_cell(pushed[0], pushed[1], p, rows, cols, score_cols):
                                        m += 1
                                        break
            return float(n), float(m)

        def mobility(board, p):
            total = 0
            if not compute_targets:
                return total
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == p and cell.side == "stone":
                        info = compute_targets(board, x, y, p, rows, cols, score_cols)
                        total += len(info.get('moves', set()))
            return total

        def river_reach(board, p):
            reach = 0
            if not get_flow:
                return reach
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == p and cell.side == "river":
                        flow = get_flow(board, x, y, x, y, p, rows, cols, score_cols)
                        reach += len(flow)
            return reach

        def proximity(board, p):
            srow = own_score_row(p)
            center = sum(score_cols) / len(score_cols)
            total = 0.0
            maxd = rows + cols
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == p and cell.side == "stone":
                        d = abs(y - srow) + abs(x - center)
                        total += (maxd - d)
            return total

        def disruption(board, mep):
            blocked = 0
            op = opp if mep == me else me
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == op and cell.side == "river":
                        dirs = [(1, 0), (-1, 0)] if cell.orientation == "horizontal" else [(0, 1), (0, -1)]
                        for dx, dy in dirs:
                            nx, ny = x + dx, y + dy
                            if in_bounds(nx, ny, rows, cols):
                                c2 = board[ny][nx]
                                if c2 and c2.owner == mep and c2.side == "stone":
                                    blocked += 1
            return blocked

        def forward_pressure(board, p):
            frontier = rows // 2
            target_row = own_score_row(p)
            pressure = 0.0
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == p and cell.side == "stone":
                        if (p == "circle" and y < frontier) or (p == "square" and y > frontier):
                            pressure += 1.5
                        distance = abs(y - target_row)
                        pressure += max(0, 4 - distance) * 0.4
                        if x in score_cols:
                            pressure += 0.5
            return pressure

        def river_barrier(board, p):
            own_row = own_score_row(p)
            barrier = 0.0
            for y in range(rows):
                for x in range(cols):
                    cell = board[y][x]
                    if cell and cell.owner == p and cell.side == "river":
                        if abs(y - own_row) <= 1:
                            barrier += 1.2
                        if x in score_cols:
                            barrier += 0.6
            return barrier

        def evaluator(board):
            phase = 0 if self.turn < 10 else (1 if self.turn < 35 else 2)
            w_c1 = [850, 1400, 2600][phase]
            w_mob = [24, 18, 12][phase]
            w_rivr = [32, 24, 14][phase]
            w_prox = [12, 10, 6][phase]
            w_disr = [24, 28, 18][phase]
            w_press = [26, 34, 42][phase]
            w_barrier = [18, 22, 28][phase]

            n_me, m_me = nm_values(board, me)
            n_op, m_op = nm_values(board, opp)
            c1 = (n_me + m_me / 10.0) - (n_op + m_op / 10.0)
            v = 0.0
            v += w_c1 * c1
            v += w_mob * (mobility(board, me) - mobility(board, opp))
            v += w_rivr * (river_reach(board, me) - river_reach(board, opp))
            v += w_prox * (proximity(board, me) - proximity(board, opp))
            v += w_disr * (disruption(board, me) - disruption(board, opp))
            v += w_press * (forward_pressure(board, me) - forward_pressure(board, opp))
            v += w_barrier * (river_barrier(board, me) - river_barrier(board, opp))
            return v

        return evaluator

    def _order_moves(self, moves: List[Dict[str, Any]], board: List[List[Any]], rows: int, cols: int, score_cols: List[int], player: Optional[str]=None) -> List[Dict[str, Any]]:
        if player is None:
            player = self.player
        own_row = top_score_row() if player == "circle" else bottom_score_row(rows)
        opp_row = bottom_score_row(rows) if player == "circle" else top_score_row()

        def prio(m: Dict[str, Any]) -> int:
            p = 0
            if m.get("to_score"):
                p += 5000
            if m.get("exit_score"):
                p -= 6000
            fx, fy = m.get("from", [0, 0])
            if m["action"] == "move" and "to" in m:
                tx, ty = m["to"]
                if ty == own_row and tx in score_cols:
                    p += 2000
                if m.get("via_river"):
                    depth = (fy - ty) if player == "square" else (ty - fy)
                    p += 600 + max(0, depth) * 12
                if (player == "circle" and ty > fy) or (player == "square" and ty < fy):
                    p += 120
            if m["action"] == "push":
                piece = board[fy][fx]
                if piece and piece.side == "river":
                    p += 900
                else:
                    p += 350
                if m.get("to_score"):
                    p += 2200
            if m["action"] == "flip" and "orientation" in m:
                if m["orientation"] == "vertical":
                    if 0 < len(score_cols):
                        center = sum(score_cols) / len(score_cols)
                        dist = abs(fx - center)
                        p += max(0, 160 - int(dist) * 25)
                    else:
                        p += 120
                if m.get("from_score"):
                    p -= 800
            if m["action"] == "rotate":
                p += 30
                if m.get("from_score"):
                    p -= 600
            return p

        moves.sort(key=prio, reverse=True)
        return moves
