import argparse, json, copy, time
from typing import List, Optional, Dict, Any, Tuple

# Agent factory now expects only (side, strategy)
from agent import get_agent

# pygame optional
try:
    import pygame
except Exception:
    pygame = None

# ---------------- Config ----------------
DEFAULT_ROWS = 13
DEFAULT_COLS = 12
CELL = 48
MARGIN = 60
FPS = 30
TIME_PER_PLAYER = 10 * 60
WIN_COUNT = 4

# Colors
BG = (30,30,30)
BOARD_COLOR = (116,185,255)
GRID_COLOR = (99,110,114)
HIGHLIGHT = (46,213,115)
CIRCLE_COLOR = (255,107,107)
SQUARE_COLOR = (72,219,251)
STONE_FILL = (245,246,250)
RIVER_FILL = (60,140,200)
TEXT_COLOR = (245,245,245)

# ---------------- Piece & Board Utilities ----------------
def opponent(p):
    return 'circle' if p == 'square' else 'square'
class Piece:
    def __init__(self, owner:str, side:str="stone", orientation:Optional[str]=None):
        self.owner = owner
        self.side = side
        self.orientation = orientation if orientation else "horizontal"
    def copy(self): return Piece(self.owner, self.side, self.orientation)
    def to_dict(self): return {"owner":self.owner,"side":self.side,"orientation":self.orientation}
    @staticmethod
    def from_dict(d:Optional[Dict[str,Any]]):
        if d is None: return None
        return Piece(d["owner"], d.get("side","stone"), d.get("orientation","horizontal"))

def empty_board(rows:int, cols:int) -> List[List[Optional[Piece]]]:
    return [[None for _ in range(cols)] for __ in range(rows)]

def default_start_board(rows:int, cols:int) -> List[List[Optional[Piece]]]:
    board = empty_board(rows, cols)
    width = min(6, max(2, cols - 6))
    start_cols = list(range((cols - width)//2, (cols - width)//2 + width))
    top_rows = [3,4]   # buffer at 0
    bot_rows = [rows-5, rows-4]  # buffer at rows-1
    for r in top_rows:
        for c in start_cols:
            board[r][c] = Piece("square","stone")
    for r in bot_rows:
        for c in start_cols:
            board[r][c] = Piece("circle","stone")
    return board

def load_board_from_file(path:str):
    with open(path,"r",encoding="utf-8") as fh:
        data = json.load(fh)
    raw = data.get("board")
    rows = len(raw); cols = len(raw[0])
    board = [[Piece.from_dict(cell) if cell else None for cell in row] for row in raw]
    return board, rows, cols

def save_board_to_file(board, path:str):
    data = {"board":[[cell.to_dict() if cell else None for cell in row] for row in board]}
    with open(path,"w",encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ---------------- Score helpers ----------------
def score_cols_for(cols:int) -> List[int]:
    w=4
    start = max(0, (cols - w)//2)
    return list(range(start, start+w))

def top_score_row() -> int:
    return 2

def bottom_score_row(rows:int) -> int:
    return rows - 3

def in_bounds(x:int,y:int,rows:int,cols:int) -> bool:
    return 0 <= x < cols and 0 <= y < rows

def is_opponent_score_cell(x:int,y:int,player:str,rows:int,cols:int,score_cols:List[int]) -> bool:
    if player == "circle":
        return (y == bottom_score_row(rows)) and (x in score_cols)
    else:
        return (y == top_score_row()) and (x in score_cols)

# ---------------- River flow & validation (authoritative) ----------------
def get_river_flow_destinations(board:List[List[Optional[Piece]]],
                                rx:int, ry:int, sx:int, sy:int, player:str,
                                rows:int, cols:int, score_cols:List[int],
                                river_push:bool=False) -> List[Tuple[int,int]]:
    destinations=[]; visited=set(); queue=[(rx,ry)]
    while queue:
        x,y = queue.pop(0)
        if (x,y) in visited or not in_bounds(x,y,rows,cols): continue
        visited.add((x,y))
        cell = board[y][x]
        if river_push and x==rx and y==ry:
            cell = board[sy][sx]
        if cell is None:
            if is_opponent_score_cell(x,y,player,rows,cols,score_cols):
                # block entering opponent score
                pass
            else:
                destinations.append((x,y))
            continue
        if cell.side != "river":
            continue
        dirs = [(1,0),(-1,0)] if cell.orientation == "horizontal" else [(0,1),(0,-1)]
        for dx,dy in dirs:
            nx, ny = x+dx, y+dy
            while in_bounds(nx,ny,rows,cols):
                if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols):
                    break
                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx,ny)); nx += dx; ny += dy; continue
                if nx==sx and ny==sy:
                    nx += dx; ny += dy; continue
                if next_cell.side == "river":
                    queue.append((nx,ny)); break
                break
    out=[]; seen=set()
    for d in destinations:
        if d not in seen:
            seen.add(d); out.append(d)
    return out

# ---------------- Compute valid targets (authoritative) ----------------
def compute_valid_targets(board:List[List[Optional[Piece]]],
                          sx:int, sy:int, player:str,
                          rows:int, cols:int, score_cols:List[int]) -> Dict[str,Any]:
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
        # block entering opponent score cell
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols):
            continue
        target = board[ty][tx]
        if target is None:
            moves.add((tx,ty))
        elif target.side == "river":
            flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols)
            for d in flow: moves.add(d)
        else:
            # stone occupied
            if p.side == "stone":
                px,py = tx+dx, ty+dy
                if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,player,rows,cols,score_cols):
                    pushes.append(((tx,ty),(px,py)))
            else:
                flow = get_river_flow_destinations(board, tx, ty, sx, sy, player, rows, cols, score_cols, river_push=True)
                for d in flow:
                    if not is_opponent_score_cell(d[0],d[1],player,rows,cols,score_cols):
                        pushes.append(((tx,ty),(d[0],d[1])))
    return {'moves': moves, 'pushes': pushes}

# ---------------- Validate & apply move (authoritative) ----------------
def validate_and_apply_move(board:List[List[Optional[Piece]]],
                            move:Dict[str,Any],
                            player:str,
                            rows:int, cols:int, score_cols:List[int]) -> Tuple[bool,str]:
    if not isinstance(move, dict):
        return False, "move must be dict"
    action = move.get("action")
    if action == "move":
        fr = move.get("from"); to = move.get("to")
        if not fr or not to: return False, "move needs from & to"
        fx,fy = int(fr[0]), int(fr[1]); tx,ty = int(to[0]), int(to[1])
        if not in_bounds(fx,fy,rows,cols) or not in_bounds(tx,ty,rows,cols): return False, "oob"
        if is_opponent_score_cell(tx,ty,player,rows,cols,score_cols): return False, "can't go into opponent score"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if board[ty][tx] is None:
            board[ty][tx]=piece; board[fy][fx]=None; return True, "moved"
        pushed = move.get("pushed_to")
        if not pushed: return False, "destination occupied; pushed_to required"
        ptx,pty = int(pushed[0]), int(pushed[1])
        dx = tx - fx; dy = ty - fy
        if (ptx,pty) != (tx+dx, ty+dy): return False, "invalid pushed_to"
        if not in_bounds(ptx,pty,rows,cols): return False, "oob"
        if is_opponent_score_cell(ptx,pty,player,rows,cols,score_cols): return False, "can't push into opponent score"
        if board[pty][ptx] is not None: return False, "pushed_to not empty"
        board[pty][ptx] = board[ty][tx]; board[ty][tx] = piece; board[fy][fx] = None
        return True, "move+push applied"

    elif action == "push":
        fr = move.get("from"); to = move.get("to"); pushed = move.get("pushed_to")
        if not fr or not to or not pushed:
            return False, "push needs from,to,pushed_to"

        fx, fy = int(fr[0]), int(fr[1])
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])

        if not (in_bounds(fx,fy,rows,cols) and in_bounds(tx,ty,rows,cols) and in_bounds(px,py,rows,cols)):
            return False, "oob"

        if (is_opponent_score_cell(tx,ty,player,rows,cols,score_cols) or
            is_opponent_score_cell(px,py,player,rows,cols,score_cols)):
            return False, "push would enter opponent score cell"

        piece = board[fy][fx]
        if piece is None or piece.owner != player:
            return False, "invalid piece"

        if board[ty][tx] is None:
            return False, "to must be occupied"
        if board[py][px] is not None:
            return False, "pushed_to not empty"

        if piece.side == "river" and board[ty][tx].side == "river":
            return False, "rivers cannot push rivers"

        info = compute_valid_targets(board, fx, fy, player, rows, cols, score_cols)
        valid_pairs = info['pushes']
        if ((tx,ty), (px,py)) not in valid_pairs:
            return False, "push pair invalid"

        board[py][px] = board[ty][tx]  # enemy goes to pushed_to
        board[ty][tx] = board[fy][fx]  # mover goes into enemy’s cell
        board[fy][fx] = None           # origin cleared

        mover = board[ty][tx]
        if mover.side == "river":
            mover.side = "stone"
            mover.orientation = None

        return True, "push applied"
    
        

        # if (tx,ty),(px,py) not in valid_pairs and not any(of==(tx,ty) and pf==(px,py) for of,pf in valid_pairs):
        #     return False, "push pair invalid"
        board[py][px] = board[ty][tx]; board[ty][tx] = board[fy][fx]; board[fy][fx] = None
        return True, "push applied"

    elif action == "flip":
        fr = move.get("from")
        if not fr: return False, "flip needs from"
        fx,fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx,fy,rows,cols): return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid piece"
        if piece.side == "stone":
            ori = move.get("orientation")
            if ori not in ("horizontal","vertical"): return False, "stone->river needs orientation"
            # check resulting river flow doesn't reach opponent score
            piece.side="river"; piece.orientation=ori
            flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
            # revert for now; we will finalize only if safe
            piece.side="stone"; piece.orientation=None
            for (dx,dy) in flow:
                if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                    return False, "flip would allow flow into opponent score"
            # commit flip
            piece.side="river"; piece.orientation=ori
            return True, "flipped to river"
        else:
            piece.side="stone"; piece.orientation=None
            return True, "flipped to stone"

    elif action == "rotate":
        fr = move.get("from")
        if not fr: return False, "rotate needs from"
        fx,fy = int(fr[0]), int(fr[1])
        if not in_bounds(fx,fy,rows,cols): return False, "oob"
        piece = board[fy][fx]
        if piece is None or piece.owner != player: return False, "invalid"
        if piece.side != "river": return False, "rotate only on river"
        piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
        flow = get_river_flow_destinations(board, fx, fy, fx, fy, player, rows, cols, score_cols)
        for (dx,dy) in flow:
            if is_opponent_score_cell(dx,dy,player,rows,cols,score_cols):
                piece.orientation = "horizontal" if piece.orientation=="vertical" else "vertical"
                return False, "rotation allows flow into opponent score"
        return True, "rotated"

    return False, "unknown action"

# ---------------- Generate moves for agents (compatibility) ----------------
def generate_all_moves(board:List[List[Optional[Piece]]],
                       player:str, rows:int, cols:int, score_cols:List[int]) -> List[Dict[str,Any]]:
    # This is a convenience implementation; agents have their own generators,
    # but main provides this as well for reference or alternative usage.
    moves=[]
    dirs=[(1,0),(-1,0),(0,1),(0,-1)]
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p or p.owner != player: continue
            if p.side == "stone":
                for dx,dy in dirs:
                    nx,ny = x+dx,y+dy
                    if not in_bounds(nx,ny,rows,cols): continue
                    if is_opponent_score_cell(nx,ny,player,rows,cols,score_cols): continue
                    if board[ny][nx] is None:
                        moves.append({"action":"move","from":[x,y],"to":[nx,ny]})
                    else:
                        if board[ny][nx].owner != player:
                            px,py = nx+dx, ny+dy
                            if in_bounds(px,py,rows,cols) and board[py][px] is None and not is_opponent_score_cell(px,py,player,rows,cols,score_cols):
                                moves.append({"action":"push","from":[x,y],"to":[nx,ny],"pushed_to":[px,py]})
                # flips: only add flips that are safe (flow won't reach opponent score)
                for ori in ("horizontal","vertical"):
                    p.side="river"; p.orientation=ori
                    flow = get_river_flow_destinations(board, x, y, x, y, player, rows, cols, score_cols)
                    p.side="stone"; p.orientation=None
                    if not any(is_opponent_score_cell(dx,dy,player,rows,cols,score_cols) for dx,dy in flow):
                        moves.append({"action":"flip","from":[x,y],"orientation":ori})
            else:
                moves.append({"action":"flip","from":[x,y]})
                # rotate if safe
                new_ori = "vertical" if p.orientation=="horizontal" else "horizontal"
                p.orientation = new_ori
                flow = get_river_flow_destinations(board, x, y, x, y, player, rows, cols, score_cols)
                p.orientation = "horizontal" if new_ori=="vertical" else "vertical"
                if not any(is_opponent_score_cell(dx,dy,player,rows,cols,score_cols) for dx,dy in flow):
                    moves.append({"action":"rotate","from":[x,y]})
    return moves

# ---------------- Win check ----------------
def check_win(board:List[List[Optional[Piece]]], rows:int, cols:int, score_cols:List[int]) -> Optional[str]:
    top = top_score_row(); bot = bottom_score_row(rows)
    ccount=0; scount=0
    for x in score_cols:
        if in_bounds(x, top, rows, cols):
            p = board[top][x]; 
            if p and p.owner=="circle" and p.side=="stone": ccount+=1
        if in_bounds(x, bot, rows, cols):
            q = board[bot][x]
            if q and q.owner=="square" and q.side=="stone": scount+=1
    if ccount >= WIN_COUNT: return "circle"
    if scount >= WIN_COUNT: return "square"
    return None

# ---------------- ASCII for CLI ----------------
def board_to_ascii(board:List[List[Optional[Piece]]], rows:int, cols:int) -> str:
    rows_out=[]
    for y in range(rows):
        row=[]
        for x in range(cols):
            p = board[y][x]
            if not p: row.append(".")
            else:
                ch = "C" if p.owner=="circle" else "S"
                row.append(ch.lower() if p.side=="river" else ch)
        rows_out.append(" ".join(row))
    return "\n".join(rows_out)

# ---------------- GUI rendering & loop ----------------
if pygame:
    pygame.init()
    FONT = pygame.font.SysFont("arial", 14)
    BIGFONT = pygame.font.SysFont("arial", 18)

def draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current):
    screen.fill(BG)
    # background
    board_rect = pygame.Rect(MARGIN-20, MARGIN-20, cols*CELL+40, rows*CELL+40)
    pygame.draw.rect(screen, BOARD_COLOR, board_rect, border_radius=10)
    # draw exact scoring cells (centered 4)
    top = top_score_row(); bot = bottom_score_row(rows)
    for x in score_cols:
        r = pygame.Rect(MARGIN + x*CELL - CELL//2, MARGIN + top*CELL - CELL//2, CELL, CELL)
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA); s.fill((*CIRCLE_COLOR,80)); screen.blit(s, r.topleft)
    for x in score_cols:
        r = pygame.Rect(MARGIN + x*CELL - CELL//2, MARGIN + bot*CELL - CELL//2, CELL, CELL)
        s = pygame.Surface((r.w, r.h), pygame.SRCALPHA); s.fill((*SQUARE_COLOR,80)); screen.blit(s, r.topleft)

    # grid points
    for y in range(rows):
        for x in range(cols):
            cx = MARGIN + x*CELL; cy = MARGIN + y*CELL
            pygame.draw.circle(screen, GRID_COLOR, (cx,cy), 3)

    # highlights
    for hx,hy in highlights:
        pygame.draw.circle(screen, HIGHLIGHT, (MARGIN + hx*CELL, MARGIN + hy*CELL), 18, 3)

    # draw pieces
    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p: continue
            cx = MARGIN + x*CELL; cy = MARGIN + y*CELL
            color = CIRCLE_COLOR if p.owner=="circle" else SQUARE_COLOR
            pygame.draw.circle(screen, color, (cx,cy), CELL//2 - 6)
            if p.side == "stone":
                pygame.draw.circle(screen, STONE_FILL, (cx,cy), CELL//2 - 14)
                pygame.draw.circle(screen, color, (cx,cy), CELL//2 - 14, 2)
            else:
                if p.orientation == "horizontal":
                    pygame.draw.rect(screen, RIVER_FILL, (cx - (CELL//2 - 12), cy - 6, CELL - 24, 12), border_radius=6)
                else:
                    pygame.draw.rect(screen, RIVER_FILL, (cx - 6, cy - (CELL//2 - 12), 12, CELL - 24), border_radius=6)

    # message & timers
    msg_surf = BIGFONT.render(msg, True, TEXT_COLOR)
    screen.blit(msg_surf, (20, rows*CELL + MARGIN - 6))
    t1 = FONT.render(f"Circle: {format_time(timers['circle'])}", True, CIRCLE_COLOR)
    t2 = FONT.render(f"Square: {format_time(timers['square'])}", True, SQUARE_COLOR)
    turn = FONT.render(f"Turn: {current.title()}", True, TEXT_COLOR)
    screen.blit(t1, (20, 10)); screen.blit(t2, (20, 30)); screen.blit(turn, (20, 50))
    pygame.display.flip()

def format_time(sec:float) -> str:
    if sec < 0: sec = 0
    m = int(sec//60); s = int(sec%60)
    return f"{m:02d}:{s:02d}"

def run_gui(mode:str, circle_strategy:str, square_strategy:str, load_file:Optional[str], rows:int, cols:int):
    if not pygame:
        print("pygame not available; use --nogui")
        return
    score_cols = score_cols_for(cols)
    # if load_file:
    #     board, r_loaded, c_loaded = load_board_from_file(load_file)
    #     rows, cols = r_loaded, c_loaded
    # else:
    board = default_start_board(rows, cols)
    screen = pygame.display.set_mode((cols*CELL + MARGIN*2, rows*CELL + MARGIN*2 + 80))
    pygame.display.set_caption("Final Ultimate")
    clock = pygame.time.Clock()
    players = {"circle":"human","square":"human"}
    if mode == "hvai": players["square"]="ai"
    elif mode == "aivai": players = {"circle":"ai","square":"ai"}
    # instantiate agents (they only receive board)
    agent_circle = get_agent("circle", circle_strategy)
    agent_square = get_agent("square", square_strategy)
    agents = {}
    if players["circle"]=="ai": agents["circle"] = agent_circle
    if players["square"]=="ai": agents["square"] = agent_square

    timers = {"circle": TIME_PER_PLAYER, "square": TIME_PER_PLAYER}
    last = time.time()
    current = "circle"
    selected = None
    highlights = set()
    msg = "Click piece to select. M:Move P:Push F:Flip R:Rotate. H/V for orientation in flip mode."
    action_mode = None
    winner = None
    push_stage = None
    push_candidate = None

    while True:
        dt = clock.tick(FPS)/1000.0
        now = time.time()
        if not winner:
            timers[current] -= (now - last); last = now
            if timers[current] <= 0:
                winner = opponent(current); msg = f"{current.title()} timed out. {winner.title()} wins!"
        # AI turn (single call)
        if players[current] == "ai" and not winner:
            agent = agents[current]
            move = agent.choose(board, rows, cols, score_cols)
            if move:
                ok, info = validate_and_apply_move(board, move, current, rows, cols, score_cols)
                msg = f"AI {current}: {info}"
                if ok:
                    w = check_win(board, rows, cols, score_cols)
                    if w: winner = w; msg = f"{w.title()} wins!"
                    current = opponent(current); selected=None; highlights=set(); action_mode=None; push_stage=None; push_candidate=None
                else:
                    current = opponent(current)
            else:
                current = opponent(current)
            draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current)
            continue

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_s:
                    save_board_to_file(board, "saved_board.json"); msg = "Saved board"
                if ev.key == pygame.K_ESCAPE:
                    selected=None; highlights=set(); action_mode=None; push_stage=None; push_candidate=None; msg="Cleared"
                if selected and ev.key == pygame.K_m:
                    action_mode = "move"
                    info = compute_valid_targets(board, selected[0], selected[1], current, rows, cols, score_cols)
                    highlights = set(info['moves'])
                    msg = "Move mode: click a highlighted dest"
                if selected and ev.key == pygame.K_p:
                    action_mode="push"; push_stage=0; push_candidate=None
                    info = compute_valid_targets(board, selected[0], selected[1], current, rows, cols, score_cols)
                    own_finals = set([of for of,pf in info['pushes']])
                    highlights = set(own_finals); msg = "Push mode: click own_final"
                if selected and ev.key == pygame.K_f:
                    action_mode="flip"; msg = "Flip: press H/V for stone->river or F to flip river->stone"
                if selected and ev.key == pygame.K_r:
                    sx,sy = selected; p = board[sy][sx]
                    if p and p.owner==current and p.side=="river":
                        m = {"action":"rotate","from":[sx,sy]}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"
                            current = opponent(current); selected=None; highlights=set(); action_mode=None
                    else:
                        msg = "Rotate needs selected river piece"
                if action_mode=="flip" and selected:
                    sx,sy = selected
                    if ev.key == pygame.K_h or ev.key == pygame.K_v:
                        ori = "horizontal" if ev.key==pygame.K_h else "vertical"
                        m={"action":"flip","from":[sx,sy],"orientation":ori}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"
                            current = opponent(current); selected=None; highlights=set(); action_mode=None
                    elif ev.key == pygame.K_f:
                        m={"action":"flip","from":[sx,sy]}
                        ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols)
                        msg = info
                        if ok:
                            w = check_win(board, rows, cols, score_cols)
                            if w: winner=w; msg = f"{w.title()} wins!"
                            current = opponent(current); selected=None; highlights=set(); action_mode=None

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button==1:
                mx,my = ev.pos
                rx = round((mx - MARGIN)/CELL); ry = round((my - MARGIN)/CELL)
                if not in_bounds(rx,ry,rows,cols): continue
                if selected is None:
                    p = board[ry][rx]
                    if p and p.owner==current:
                        selected=(rx,ry); highlights=set(); action_mode=None; push_stage=None; push_candidate=None; msg=f"Selected {selected}"
                    else:
                        msg = "Select one of your pieces"
                else:
                    sx,sy = selected
                    if action_mode=="move":
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        if (rx,ry) not in info['moves']:
                            msg = "Invalid target"
                        else:
                            if board[ry][rx] is None:
                                m = {"action":"move","from":[sx,sy],"to":[rx,ry]}
                            else:
                                dx = rx - sx; dy = ry - sy
                                m = {"action":"move","from":[sx,sy],"to":[rx,ry],"pushed_to":[rx+dx,ry+dy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols); msg = info
                            if ok:
                                w = check_win(board, rows, cols, score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"
                                current = opponent(current); selected=None; highlights=set(); action_mode=None; push_stage=None; push_candidate=None
                    elif action_mode=="push":
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        push_pairs = info['pushes']
                        own_finals = set([of for of,pf in push_pairs])
                        if push_stage==0 or push_stage is None:
                            if (rx,ry) not in own_finals:
                                msg = "Click a valid own_final"
                            else:
                                push_candidate=(rx,ry)
                                pushed_options=[pf for of,pf in push_pairs if of==push_candidate]
                                highlights=set(pushed_options); push_stage=1; msg=f"Selected own_final {push_candidate}. Click pushed_to"
                        else:
                            if push_candidate is None:
                                msg = "Push error; reselect"
                                push_stage=None; push_candidate=None; highlights=set(); action_mode=None
                            else:
                                candidate_pair = (push_candidate,(rx,ry))
                                if candidate_pair not in push_pairs:
                                    msg = "Invalid pushed_to"
                                else:
                                    m={"action":"push","from":[sx,sy],"to":[push_candidate[0],push_candidate[1]],"pushed_to":[rx,ry]}
                                    ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols); msg = info
                                    push_stage=None; push_candidate=None; highlights=set(); action_mode=None
                                    if ok:
                                        w = check_win(board, rows, cols, score_cols)
                                        if w: winner=w; msg=f"{w.title()} wins!"
                                        current = opponent(current); selected=None
                    elif action_mode=="flip":
                        p = board[sy][sx]
                        if p.side=="river":
                            m={"action":"flip","from":[sx,sy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols); msg = info
                            if ok:
                                w = check_win(board, rows, cols, score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"
                                current = opponent(current); selected=None; action_mode=None
                        else:
                            msg = "Press H/V for stone->river in flip mode"
                    else:
                        info = compute_valid_targets(board,sx,sy,current,rows,cols,score_cols)
                        if (rx,ry) in info['moves']:
                            if board[ry][rx] is None:
                                m={"action":"move","from":[sx,sy],"to":[rx,ry]}
                            else:
                                dx = rx - sx; dy = ry - sy
                                m={"action":"move","from":[sx,sy],"to":[rx,ry],"pushed_to":[rx+dx,ry+dy]}
                            ok,info = validate_and_apply_move(board,m,current,rows,cols,score_cols); msg = info
                            if ok:
                                w = check_win(board, rows, cols, score_cols)
                                if w: winner=w; msg=f"{w.title()} wins!"
                                current = opponent(current); selected=None; highlights=set(); action_mode=None
                        else:
                            newp = board[ry][rx]
                            if newp and newp.owner==current:
                                selected=(rx,ry); highlights=set(); action_mode=None; msg=f"Selected {selected}"
                            else:
                                msg = "Invalid click"
        draw_board(screen, board, rows, cols, score_cols, selected, highlights, msg, timers, current)
        if winner:
            time.sleep(0.1)

# ---------------- CLI interactive runner ----------------
def run_cli(mode:str, circle_strategy:str, square_strategy:str, load_file:Optional[str], rows:int, cols:int):
    score_cols = score_cols_for(cols)
    # if load_file:
    #     board, r_loaded, c_loaded = load_board_from_file(load_file)
    #     rows, cols = r_loaded, c_loaded
    # else:
    board = default_start_board(rows, cols)
    agent_circle = get_agent("circle", circle_strategy)
    agent_square = get_agent("square", square_strategy)
    players = {"circle":"human","square":"human"}
    if mode=="hvai": players["square"]="ai"
    elif mode=="aivai": players={"circle":"ai","square":"ai"}
    current="circle"; winner=None; turn=0
    while True:
        # print("\nBoard:")
        # print(board_to_ascii(board, rows, cols))
        w = check_win(board, rows, cols, score_cols)
        if w:
            print("Winner:", w); break
        if players[current]=="ai":
            agent = agent_circle if current=="circle" else agent_square
            move = agent.choose(board, rows, cols, score_cols)
            if move is None:
                print(f"AI {current} no moves; pass"); current = opponent(current); continue
            ok,msg = validate_and_apply_move(board, move, current, rows, cols, score_cols)
            print(f"AI {current} -> {move} : {msg}")
            if not ok:
                current = opponent(current); continue
        else:
            s = input(f"{current} move JSON (or q): ").strip()
            if s.lower()=="q": break
            try:
                move = json.loads(s)
            except Exception as e:
                print("Bad JSON:", e); continue
            ok,msg = validate_and_apply_move(board, move, current, rows, cols, score_cols)
            print(msg)
            if not ok: continue
        current = opponent(current)
        turn += 1
        if turn > 2000:
            print("Turn limit -> draw"); break

# ---------------- Entrypoint ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hvh","hvai","aivai"], default="hvai")
    ap.add_argument("--circle", choices=["random","best"], default="best")
    ap.add_argument("--square", choices=["random","best"], default="random")
    ap.add_argument("--load", default=None)
    ap.add_argument("--nogui", action="store_true")
    args = ap.parse_args()

    rows = DEFAULT_ROWS; cols = DEFAULT_COLS

    if args.nogui:
        run_cli(args.mode, args.circle, args.square, args.load, rows, cols)
    else:
        run_gui(args.mode, args.circle, args.square, args.load, rows, cols)

if __name__=="__main__":
    main()

