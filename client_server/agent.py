
import random
import copy
from typing import List, Dict, Any, Optional, Tuple

# Reuse the same Piece shape as in final_ultimate.py (expects objects with .owner, .side, .orientation)
# Agents operate on the board objects directly.

def in_bounds(x:int,y:int,rows:int,cols:int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def score_cols_for(cols:int) -> List[int]:
    w = 4
    start = max(0, (cols - w)//2)
    return list(range(start, start + w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows:int) -> int:
    return rows - 3

def is_opponent_score_cell(x:int,y:int,player:str,rows:int,cols:int,score_cols:List[int]) -> bool:
    # True if (x,y) is in the opponent's scoring cell set
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

# River flow exploration used by agents — prevents entering opponent score cells.
def agent_river_flow(board, rx:int, ry:int, sx:int, sy:int, player:str, rows:int, cols:int, score_cols:List[int], river_push:bool=False) -> List[Tuple[int,int]]:
    destinations=[]
    visited=set()
    queue=[(rx,ry)]
    while queue:
        x,y = queue.pop(0)
        if (x,y) in visited or not in_bounds(x,y,rows,cols):
            continue
        visited.add((x,y))
        cell = board[y][x]
        if river_push and x==rx and y==ry:
            cell = board[sy][sx]
        if cell is None:
            if is_opponent_score_cell(x,y,player,rows,cols,score_cols):
                # block entering opponent score cell
                pass
            else:
                destinations.append((x,y))
            continue
        if getattr(cell,"side","stone") != "river":
            continue
        dirs = [(1,0),(-1,0)] if cell.orientation=="horizontal" else [(0,1),(0,-1)]
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            while in_bounds(nx,ny,rows,cols):
                if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols):
                    break
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx,ny))
                    nx += dx; ny += dy
                    continue
                if nx==sx and ny==sy:
                    nx += dx; ny += dy
                    continue
                if getattr(next_cell,"side","stone") == "river":
                    queue.append((nx,ny)); break
                break
    # unique
    out=[]; seen=set()
    for d in destinations:
        if d not in seen:
            seen.add(d); out.append(d)
    return out

def agent_compute_valid(board, sx:int, sy:int, player:str, rows:int, cols:int, score_cols:List[int]) -> Dict[str,Any]:
    if not in_bounds(sx,sy,rows,cols):
        return {'moves': set(), 'pushes': []}
    p = board[sy][sx]
    if p is None or p.owner != player:
        return {'moves': set(), 'pushes': []}
    moves=set(); pushes=[]
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in dirs:
        tx,ty = sx+dx, sy+dy
        if not in_bounds(tx,ty,rows,cols): continue
        # block moving into opponent score cell
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols):
            continue
        target = board[ty][tx]
        if target is None:
            moves.add((tx,ty))
        elif getattr(target,"side","stone") == "river":
            flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for d in flow:
                moves.add(d)
        else:
            # occupied by stone
            if getattr(p,"side","stone") == "stone":
                px,py = tx+dx, ty+dy
                if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,player,rows,cols,score_cols):
                    pushes.append(((tx,ty),(px,py)))
            else:
                flow = agent_river_flow(board, tx, ty, sx, sy, player, rows, cols, score_cols, river_push=True)
                for d in flow:
                    if not is_opponent_score_cell(d[0],d[1],player,rows,cols,score_cols):
                        pushes.append(((tx,ty),(d[0],d[1])))
    return {'moves': moves, 'pushes': pushes}

def agent_apply(board, move:Dict[str,Any], player:str, rows:int, cols:int, score_cols:List[int]) -> Tuple[bool,str]:
    # returns (ok,msg). modifies supplied board copy.
    action = move.get("action")
    if action == "move":
        fr = move.get("from"); to = move.get("to")
        if not fr or not to: return False, "bad move format"
        fx,fy = int(fr[0]), int(fr[1]); tx,ty = int(to[0]), int(to[1])
        if not in_bounds(fx,fy,rows,cols) or not in_bounds(tx,ty,rows,cols): return False, "oob"
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols): return False, "can't move into opponent score cell"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if board[ty][tx] is None:
            board[ty][tx] = piece; board[fy][fx] = None; return True, "moved"
        pushed = move.get("pushed_to")
        if not pushed: return False, "destination occupied; pushed_to required"
        ptx,pty = int(pushed[0]), int(pushed[1])
        dx = tx - fx; dy = ty - fy
        if (ptx,pty) != (tx+dx, ty+dy): return False, "invalid pushed_to"
        if not in_bounds(ptx,pty,rows,cols): return False, "oob"
        if is_opponent_score_cell(ptx,pty,player,rows,cols,score_cols): return False, "can't push into opponent score"
        if board[pty][ptx] is not None: return False, "pushed_to not empty"
        board[pty][ptx] = board[ty][tx]; board[ty][tx] = piece; board[fy][fx] = None
        return True, "pushed"
    elif action == "push":
        fr = move.get("from"); to = move.get("to"); pushed = move.get("pushed_to")
        if not fr or not to or not pushed: return False, "bad push format"
        fx,fy=int(fr[0]),int(fr[1]); tx,ty=int(to[0]),int(to[1]); px,py=int(pushed[0]),int(pushed[1])
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols) or is_opponent_score_cell(px,py,player,rows,cols,score_cols):
            return False, "push would move into opponent score cell"
        if not (in_bounds(fx,fy,rows,cols) and in_bounds(tx,ty,rows,cols) and in_bounds(px,py,rows,cols)):
            return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if board[ty][tx] is None: return False, "'to' must be occupied"
        if board[py][px] is not None: return False, "pushed_to not empty"
        board[py][px] = board[ty][tx]; board[ty][tx] = board[fy][fx]; board[fy][fx] = None
        return True, "pushed"
    elif action == "flip":
        fr = move.get("from")
        if not fr: return False, "bad flip"
        fx,fy = int(fr[0]), int(fr[1])
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "bad piece"
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal","vertical"): return False, "stone->river needs orientation"
            # check whether flow from this new river immediately allows entering opponent's score cell -> block flip
            # find all flow destinations and ensure none are opponent score cells
            board[fy][fx].side = "river"; board[fy][fx].orientation = ori
            flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)
            # revert quick check; we will finalize flip only if safe
            board[fy][fx].side = "stone"; board[fy][fx].orientation = None
            for (dx,dy) in flow:
                if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                    return False, "flip would allow flow into opponent score cell"
            # apply flip
            board[fy][fx].side="river"; board[fy][fx].orientation=ori
            return True, "flipped to river"
        else:
            # river -> stone
            board[fy][fx].side="stone"; board[fy][fx].orientation=None
            return True, "flipped to stone"
    elif action == "rotate":
        fr = move.get("from")
        if not fr: return False, "bad rotate"
        fx,fy=int(fr[0]),int(fr[1])
        piece = board[fy][fx]
        if piece is None or piece.owner != player or piece.side != "river": return False, "invalid rotate"
        piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
        # check flow safety after rotate
        flow = agent_river_flow(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        for (dx,dy) in flow:
            if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                # revert rotate
                piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
                return False, "rotate would allow flow into opponent score cell"
        return True, "rotated"
    return False, "unknown action"

# ------- Agent implementations -------

class RandomAgent:
    def __init__(self, player:str):
        self.player = player

    def generate_all_moves(self, board:List[List[Any]], rows:int, cols:int, score_cols:List[int]) -> List[Dict[str,Any]]:
        moves=[]
        dirs=[(1,0),(-1,0),(0,1),(0,-1)]
        for y in range(rows):
            for x in range(cols):
                p = board[y][x]
                if not p or p.owner != self.player: continue
                if p.side == "stone":
                    for dx,dy in dirs:
                        nx,ny = x+dx,y+dy
                        if not in_bounds(nx,ny,rows,cols): continue
                        # block destination in opponent score
                        if is_opponent_score_cell(nx,ny,self.player,rows,cols,score_cols): continue
                        if board[ny][nx] is None:
                            moves.append({"action":"move","from":[x,y],"to":[nx,ny]})
                        else:
                            if board[ny][nx].owner != self.player:
                                px,py = nx+dx, ny+dy
                                if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,self.player,rows,cols,score_cols):
                                    moves.append({"action":"push","from":[x,y],"to":[nx,ny],"pushed_to":[px,py]})
                    # flips (ensure flip safe)
                    for ori in ("horizontal","vertical"):
                        # simulate safety check using local functions
                        temp = copy.deepcopy(board)
                        temp[y][x].side="river"; temp[y][x].orientation=ori
                        flow = agent_river_flow(temp, x, y, x, y, self.player, rows, cols, score_cols)
                        if not any(is_opponent_score_cell(dx,dy,self.player,rows,cols,score_cols) for dx,dy in flow):
                            moves.append({"action":"flip","from":[x,y],"orientation":ori})
                else:
                    # river flips and rotate if safe
                    # flip back
                    moves.append({"action":"flip","from":[x,y]})
                    # rotate - only if rotation doesn't allow opponent score flow
                    new_ori = "vertical" if p.orientation=="horizontal" else "horizontal"
                    temp = copy.deepcopy(board); temp[y][x].orientation=new_ori
                    flow = agent_river_flow(temp, x, y, x, y, self.player, rows, cols, score_cols)
                    if not any(is_opponent_score_cell(dx,dy,self.player,rows,cols,score_cols) for dx,dy in flow):
                        moves.append({"action":"rotate","from":[x,y]})
        return moves

    def choose(self, board:List[List[Any]], rows:int, cols:int, score_cols:List[int]) -> Optional[Dict[str,Any]]:
        moves = self.generate_all_moves(board, rows, cols, score_cols)
        if not moves: return None
        return random.choice(moves)


class BestAgent:
    def __init__(self, player:str):
        self.player = player
        self.opp = "circle" if player=="square" else "square"

    def generate_all_moves(self, board, rows, cols, score_cols):
        # reuse RandomAgent generator logic for legal move set
        r = RandomAgent(self.player)
        return r.generate_all_moves(board, rows, cols, score_cols)

    def simulate_and_score(self, board_copy, rows, cols, score_cols):
        # simple heuristic: stones in own score cells big + progress
        sc = 0.0
        top = top_score_row()
        bot = bottom_score_row(rows)
        for y in range(rows):
            for x in range(cols):
                p = board_copy[y][x]
                if not p: continue
                if p.owner == self.player and p.side == "stone":
                    sc += 1
                    # encourage being closer to opponent side
                    if self.player == "circle":
                        sc += (rows - y) * 0.05
                    else:
                        sc += y * 0.05
                    # own scoring slot
                    if (self.player=="circle" and y==top and x in score_cols) or (self.player=="square" and y==bot and x in score_cols):
                        sc += 10
                if p.owner == self.opp and p.side=="stone":
                    sc -= 2
        return sc

    def apply_local(self, board, move, rows, cols, score_cols) -> Tuple[bool,str]:
        # use agent_apply
        bcopy = copy.deepcopy(board)
        ok,msg = agent_apply(bcopy, move, self.player, rows, cols, score_cols)
        return ok, (msg, bcopy) if ok else (msg, None)

    def choose(self, board, rows, cols, score_cols):
        moves = self.generate_all_moves(board, rows, cols, score_cols)
        if not moves: return None
        best = None; best_score = -1e9
        for m in moves:
            ok, res = self.apply_local(board, m, rows, cols, score_cols)
            if not ok: continue
            _, bcopy = res
            score = self.simulate_and_score(bcopy, rows, cols, score_cols)
            if score > best_score:
                best_score = score; best = m
        if best is None:
            return random.choice(moves)
        return best

def get_agent(side:str, strategy:str):
    s = (strategy or "random").lower()
    if s == "random":
        return RandomAgent(side)
    if s == "best":
        return BestAgent(side)
    raise ValueError("Unknown strategy")
