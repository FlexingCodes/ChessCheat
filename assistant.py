# pip install chess mss pillow numpy
import json, os, queue, threading, time, ctypes, traceback
import tkinter as tk
from tkinter import ttk, messagebox

import chess
import chess.engine
import mss
import numpy as np
from PIL import Image, ImageTk

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # coordenadas em pixel real
except Exception:
    pass

# ---- config ----
kEnginePath = r"C:\stockfish\stockfish-windows-x86-64-universal.exe"
kHintTime   = 0.25     # segundos por análise (bullet: 0.15~0.3)
kHintDepth  = 18
kHintLines  = 1        # 1 = mais rápido; 3 = mostra alternativas
kThreads    = max(1, (os.cpu_count() or 4) - 1)
kHashMb     = 256
kScanMs     = 40       # intervalo do scan
kSettleMs   = 80       # leitura precisa ficar igual por esse tempo
kResyncMs   = 600      # posição sem lance legal: reconstrói pela imagem depois disso
kHotDist    = 45       # diferença de cor pra considerar casa destacada (último lance)
kCell       = 32       # px por casa depois do downscale
kMargin     = 0.15     # corta a borda da casa (coordenadas, destaques)
kDarkThresh = 70       # pixel < isso = contorno/peça preta
kEmptyMin   = 0.015    # menos que isso de pixels escuros = casa vazia
kMaxDist    = 0.22     # distância máx pro template casar
kCfgFile    = "assistente_cfg.json"
kSqPx       = 50
kLightSq, kDarkSq = "#e4e6ee", "#6f7f9c"   # mini tabuleiro (cores diferentes das do site de propósito)
kOverlay    = True     # marcação em cima do site (liga/desliga também pela GUI)
kKey        = "#010203"  # cor-chave transparente do overlay
kFromCol    = "#00c8ff"  # moldura da peça que sai
kToCol      = "#ff9d00"  # moldura da casa de destino
kArrowCol   = "#ff9d00"
kGridCol    = "#00ff88"  # grade de confirmação do auto-detect
kAutoFollow = True     # re-detecta o tabuleiro sozinho se ele sumir/mexer
kAutoColors = 12       # quantas cores mais frequentes testar na detecção
kAutoMinPx  = 4000     # cor com menos pixels que isso é ignorada
kAutoScore  = 0.85     # quão xadrez (alternado) a grade precisa ser
# ----------------

kCrop = int(kCell * kMargin)
kInner = kCell - 2 * kCrop


def grab(sct, region):
    s = sct.grab(region)
    return Image.frombytes("RGB", s.size, s.bgra, "raw", "BGRX")


def longest_run(proj, thr):
    on = np.concatenate(([False], proj > thr, [False])).astype(np.int8)
    d = np.diff(on)
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    if len(starts) == 0:
        return 0, 0
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def detect_board(sct, excl=None):
    """Acha o tabuleiro (8x8 alternando duas cores chapadas) no monitor principal.
    excl = (x, y, w, h) em coords de tela a ignorar (a janela do próprio assistente)."""
    mon = sct.monitors[1]
    shot = sct.grab(mon)
    a = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))
    if excl:
        ex, ey, ew, eh = excl
        y0, x0 = max(0, ey - mon["top"]), max(0, ex - mon["left"])
        a[y0:max(y0, ey - mon["top"] + eh), x0:max(x0, ex - mon["left"] + ew)] = 0
    p = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
    vals, cnt = np.unique(p, return_counts=True)
    order = np.argsort(-cnt)[:kAutoColors]
    cols = [int(vals[i]) for i in order if cnt[i] >= kAutoMinPx]
    masks = [p == c for c in cols]
    par = np.where((np.add.outer(np.arange(8), np.arange(8)) % 2) == 0, 1, -1)
    best = None
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            U = masks[i] | masks[j]
            cs, rs = U.sum(axis=0), U.sum(axis=1)
            if cs.max() < 160 or rs.max() < 160:
                continue
            xa, xb = longest_run(cs, 0.3 * cs.max())
            ya, yb = longest_run(rs, 0.3 * rs.max())
            w, h = xb - xa, yb - ya
            if min(w, h) < 160 or abs(w - h) > 0.03 * max(w, h):
                continue
            side = (w + h) // 2
            cell = side / 8
            sg = np.zeros((8, 8))
            for r in range(8):
                for c in range(8):
                    ys, ye = int(ya + (r + .15) * cell), int(ya + (r + .85) * cell)
                    xs, xe = int(xa + (c + .15) * cell), int(xa + (c + .85) * cell)
                    n1 = masks[i][ys:ye, xs:xe].sum()
                    n2 = masks[j][ys:ye, xs:xe].sum()
                    if n1 + n2 > 0.1 * (ye - ys) * (xe - xs):
                        sg[r, c] = 1 if n1 > n2 else -1
            used = int((sg != 0).sum())
            if used < 40 or abs((sg * par).sum()) / used < kAutoScore:
                continue
            if best is None or side > best["width"]:
                best = {"left": mon["left"] + xa, "top": mon["top"] + ya, "width": side, "height": side}
    return best


def sq_to_cell(sq, wb):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return (7 - r, f) if wb else (r, 7 - f)


def cell_to_sq(r, c, wb):
    r, c = int(r), int(c)
    return chess.square(c, 7 - r) if wb else chess.square(7 - c, r)


def cell_masks(img):
    """(8, 8, kInner, kInner): 1.0 onde o pixel é escuro (contorno/peça preta)."""
    g = np.asarray(img.convert("L").resize((8 * kCell, 8 * kCell), Image.BILINEAR))
    g = g.reshape(8, kCell, 8, kCell).transpose(0, 2, 1, 3)  # (linha, coluna, y, x)
    g = g[:, :, kCrop:kCell - kCrop, kCrop:kCell - kCrop]
    return (g < kDarkThresh).astype(np.float32)


def calibrate_templates(img, wb):
    """Aprende as peças da imagem. A posição pode ser a inicial ou ter até 2 lances
    (preto: o bot já jogou). Testa todas e fica com a que deixa as peças mais consistentes."""
    flat = cell_masks(img).reshape(8, 8, -1)
    has = flat.mean(axis=2) >= kEmptyMin
    start = chess.Board()
    cands = [start]
    for m1 in list(start.legal_moves):
        b1 = start.copy()
        b1.push(m1)
        cands.append(b1)
        for m2 in list(b1.legal_moves):
            b2 = b1.copy()
            b2.push(m2)
            cands.append(b2)
    best = (1e18, None, None)
    for b in cands:
        groups, pen = {}, 0.0
        for sq in chess.SQUARES:
            r, c = sq_to_cell(sq, wb)
            p = b.piece_at(sq)
            if p:
                groups.setdefault(p.symbol(), []).append(flat[r, c])
            if bool(p) != bool(has[r, c]):
                pen += 1.0
        tpl, dev = {}, 0.0
        for s, v in groups.items():
            a = np.mean(v, axis=0)
            tpl[s] = a
            dev += sum(float(np.abs(x - a).mean()) for x in v)
        if dev + pen < best[0]:
            best = (dev + pen, tpl, b)
    return best


def read_board(img, tpl, wb):
    m = cell_masks(img).reshape(64, -1)
    syms = list(tpl)
    T = np.stack([tpl[s].ravel() for s in syms])               # (12, n)
    d = np.abs(m[:, None, :] - T[None, :, :]).mean(axis=2)     # (64, 12)
    best = d.argmin(axis=1)
    seen = {}
    for i in range(64):
        if m[i].mean() < kEmptyMin:
            continue
        if d[i, best[i]] < kMaxDist:
            seen[cell_to_sq(i // 8, i % 8, wb)] = syms[best[i]]
    return seen


def detect_turn(img, seen, wb):
    """De quem é a vez, pelo destaque do último lance: a peça na casa destacada
    é de quem jogou por último. None se não der pra saber."""
    rgb = np.asarray(img.convert("RGB").resize((8 * kCell, 8 * kCell), Image.BILINEAR)).astype(np.int16)
    t = rgb.reshape(8, kCell, 8, kCell, 3).transpose(0, 2, 1, 3, 4)  # (linha, coluna, y, x, rgb)
    h = kCell // 2
    patches = [t[:, :, :3, :3], t[:, :, :3, -3:], t[:, :, h - 1:h + 2, :3], t[:, :, h - 1:h + 2, -3:]]
    col = np.median(np.stack([p.mean(axis=(2, 3)) for p in patches]), axis=0)  # (8, 8, 3)
    par = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(bool)
    base = np.zeros_like(col)
    base[par] = np.median(col[par], axis=0)
    base[~par] = np.median(col[~par], axis=0)
    hot = np.abs(col - base).sum(axis=2) > kHotDist
    cells = list(zip(*np.nonzero(hot)))
    if len(cells) != 2:
        return None
    occ = [cell_to_sq(r, c, wb) for r, c in cells]
    occ = [s for s in occ if s in seen]
    if len(occ) != 1:
        return None
    return not seen[occ[0]].isupper()  # moveu peça branca -> vez das pretas


def placement(b):
    return {sq: p.symbol() for sq, p in b.piece_map().items()}


START = placement(chess.Board())


def find_moves(b, seen, depth):
    for mv in list(b.legal_moves):
        b.push(mv)
        if placement(b) == seen:
            b.pop()
            return [mv]
        if depth > 1:
            sub = find_moves(b, seen, depth - 1)
            if sub:
                b.pop()
                return [mv] + sub
        b.pop()
    return None


def sync(board, seen, mine, allow_fallback, turn_hint):
    """Casa a leitura da tela com a partida rastreada.
    Retorna: Board (ok) | False (tenta de novo depois) | None (posição inválida)."""
    if seen == START:
        return chess.Board()
    if placement(board) == seen:
        return board
    mvs = find_moves(board, seen, 2)  # até 2 plies (teu lance + resposta do bot)
    if mvs:
        for m in mvs:
            board.push(m)
        return board
    if not allow_fallback:
        return False
    # reconstrói só pela imagem (qualquer posição). vez: destaque do último lance, senão a tua
    turns = (turn_hint, not turn_hint) if turn_hint is not None else (mine, not mine)
    for turn in turns:
        nb = chess.Board(None)
        nb.set_piece_map({sq: chess.Piece.from_symbol(s) for sq, s in seen.items()})
        nb.turn = turn
        nb.castling_rights = chess.BB_CORNERS
        nb.castling_rights = nb.clean_castling_rights()
        if nb.is_valid():
            return nb
    return None


def worker(app):
    try:
        eng = chess.engine.SimpleEngine.popen_uci(kEnginePath)
        eng.configure({"Threads": kThreads, "Hash": kHashMb})
    except Exception as e:
        app.q.put(("status", f"stockfish não abriu: {e}"))
        return
    board, last, since, done, shown, bad = chess.Board(), None, time.time(), None, None, None
    lost, last_det = None, 0.0
    with mss.mss() as sct:
        while app.alive:
            time.sleep(kScanMs / 1000)
            if app.reset:
                board, last, since, done, shown, bad, app.reset = \
                    chess.Board(), None, time.time(), None, None, None, False
            try:
                now = time.time()
                # auto-detecção: sem região, ou reis sumiram por mais de 1s (tabuleiro mexeu)
                if app.auto and now - last_det > 2.0 and (not app.region or (lost and now - lost > 1.0)):
                    last_det = now
                    det = detect_board(sct, app.self_rect)
                    if det:
                        lost = None
                        if det != app.region:
                            app.region = det
                            app.q.put(("region", det))
                        board, last, since, done, shown, bad = \
                            chess.Board(), None, time.time(), None, None, None
                    else:
                        app.q.put(("status", "não achei o tabuleiro (tema com textura? usa Manual)"))
                if not (app.region and app.tpl):
                    continue
                wb = app.white_bottom
                mine = chess.WHITE if wb else chess.BLACK
                img = grab(sct, app.region)
                seen = read_board(img, app.tpl, wb)
                now = time.time()
                if seen != last:
                    last, since = seen, now
                if now - since < kSettleMs / 1000:
                    continue
                vals = list(seen.values())
                kings_ok = vals.count("K") == 1 and vals.count("k") == 1
                if kings_ok:
                    lost = None
                elif lost is None:
                    lost = now
                if seen != shown:  # o mini tabuleiro sempre espelha a leitura
                    shown = seen
                    app.q.put(("seen", dict(seen)))
                if not app.on or seen == done:
                    continue
                if not kings_ok:
                    if seen != bad:
                        bad = seen
                        app.q.put(("status", "não achei os dois reis (região/calibração?)"))
                    continue
                allow = now - since >= kResyncMs / 1000
                nb = sync(board, seen, mine, allow, detect_turn(img, seen, wb) if allow else None)
                if nb is False:
                    if seen != bad:
                        bad = seen
                        app.q.put(("status", "lendo..."))
                    continue
                done = seen
                if nb is None:
                    app.q.put(("status", "posição inválida (recalibra?)"))
                    continue
                board = nb
                if board.is_game_over():
                    app.q.put(("wait", f"fim: {board.result()}"))
                elif board.turn != mine:
                    app.q.put(("wait", "esperando o oponente..."))
                else:
                    infos = eng.analyse(board, chess.engine.Limit(time=kHintTime, depth=kHintDepth),
                                        multipv=kHintLines)
                    lines = []
                    for i in infos:
                        pv, sc = i["pv"], i["score"].pov(mine)
                        ev = f"M{sc.mate()}" if sc.is_mate() else f"{sc.score() / 100:+.2f}"
                        lines.append((pv[0].uci(), board.san(pv[0]), ev, board.variation_san(pv[:6])))
                    app.q.put(("hint", board.copy(stack=False), lines))
            except Exception as e:
                app.q.put(("status", f"erro: {e}"))
    eng.quit()


class Overlay:
    """Janela transparente, sem clique, por cima do tabuleiro. Excluída da captura de tela
    pra não entrar na leitura; sem isso (Windows antigo) desenha só molduras na borda das casas."""

    def __init__(self, root):
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=kKey)
        self.win.attributes("-transparentcolor", kKey)
        self.cv = tk.Canvas(self.win, bg=kKey, highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)
        self.win.withdraw()
        self.ready, self.excluded, self.visible = False, False, False

    def _setup(self):
        u = ctypes.windll.user32
        u.GetAncestor.restype = ctypes.c_void_p
        u.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        u.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        u.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        hwnd = u.GetAncestor(self.win.winfo_id(), 2)
        ex = u.GetWindowLongW(hwnd, -20)
        # layered | transparente ao clique | toolwindow | sem foco
        u.SetWindowLongW(hwnd, -20, ex | 0x80000 | 0x20 | 0x80 | 0x08000000)
        self.excluded = bool(u.SetWindowDisplayAffinity(hwnd, 0x11))  # WDA_EXCLUDEFROMCAPTURE
        self.ready = True

    def _place(self, region):
        w, h = region["width"], region["height"]
        self.win.geometry(f"{w}x{h}+{region['left']}+{region['top']}")
        self.win.deiconify()
        self.win.lift()
        self.win.update_idletasks()
        if not self.ready:
            self._setup()
        self.cv.delete("all")
        self.visible = True

    def grid(self, region):
        """Confirmação visual da região achada: borda + grade 8x8."""
        self._place(region)
        w = region["width"]
        cell = w / 8
        self.cv.create_rectangle(2, 2, w - 2, w - 2, outline=kGridCol, width=3)
        for i in range(1, 8):
            self.cv.create_line(i * cell, 0, i * cell, w, fill=kGridCol, width=1)
            self.cv.create_line(0, i * cell, w, i * cell, fill=kGridCol, width=1)

    def show(self, region, wb, mv):
        self._place(region)
        cv, cell = self.cv, region["width"] / 8
        lw = max(3, int(cell * 0.09))

        def box(sq, color):
            r, c = sq_to_cell(sq, wb)
            x0, y0 = c * cell + lw / 2 + 1, r * cell + lw / 2 + 1
            cv.create_rectangle(x0, y0, x0 + cell - lw - 2, y0 + cell - lw - 2, outline=color, width=lw)

        def ctr(sq):
            r, c = sq_to_cell(sq, wb)
            return (c + 0.5) * cell, (r + 0.5) * cell

        box(mv.from_square, kFromCol)
        box(mv.to_square, kToCol)
        if self.excluded:
            (x1, y1), (x2, y2) = ctr(mv.from_square), ctr(mv.to_square)
            dx, dy = x2 - x1, y2 - y1
            d = (dx * dx + dy * dy) ** 0.5 or 1.0
            k1, k2 = cell * 0.28 / d, cell * 0.22 / d
            cv.create_line(x1 + dx * k1, y1 + dy * k1, x2 - dx * k2, y2 - dy * k2,
                           width=max(4, int(cell * 0.16)), fill=kArrowCol, capstyle="round",
                           arrow=tk.LAST, arrowshape=(cell * 0.34, cell * 0.34, cell * 0.17))
        return self.excluded

    def hide(self):
        if self.visible:
            self.win.withdraw()
            self.visible = False


class App:
    def __init__(self, root):
        self.root = root
        root.title("Assistente de Xadrez")
        root.attributes("-topmost", True)
        root.configure(bg="#1b1b1b")
        self.q = queue.Queue()
        self.alive, self.on, self.reset, self.auto = True, False, False, kAutoFollow
        self.region, self.tpl, self.white_bottom = None, None, True
        self.self_rect = None  # retângulo da própria janela (excluído da detecção)
        self.last_hint = None  # (placement da posição, lance) do hint atual
        self.load()
        self.overlay = Overlay(root)

        bar = tk.Frame(root, bg="#1b1b1b")
        bar.pack(fill="x", padx=8, pady=(6, 2))
        ttk.Button(bar, text="Achar tabuleiro (auto)", command=self.auto_pick).pack(side="left", padx=2)
        ttk.Button(bar, text="Manual", command=self.pick).pack(side="left", padx=2)
        ttk.Button(bar, text="Calibrar (1x, início)", command=self.calibrate).pack(side="left", padx=2)

        bar2 = tk.Frame(root, bg="#1b1b1b")
        bar2.pack(fill="x", padx=8)
        self.side = ttk.Combobox(bar2, state="readonly", width=16,
                                 values=["Brancas embaixo", "Pretas embaixo"])
        self.side.current(0 if self.white_bottom else 1)
        self.side.bind("<<ComboboxSelected>>", self.on_side)
        self.side.pack(side="left", padx=2)
        chk = dict(bg="#1b1b1b", fg="#eee", selectcolor="#333", activebackground="#1b1b1b", activeforeground="#fff")
        self.act = tk.BooleanVar(value=False)
        tk.Checkbutton(bar2, text="Ativo (hint)", variable=self.act, command=self.on_act, **chk).pack(side="left", padx=6)
        self.ovv = tk.BooleanVar(value=kOverlay)
        tk.Checkbutton(bar2, text="Overlay no site", variable=self.ovv, command=self.on_ov, **chk).pack(side="left")

        bar3 = tk.Frame(root, bg="#1b1b1b")
        bar3.pack(fill="x", padx=8)
        self.autov = tk.BooleanVar(value=kAutoFollow)
        tk.Checkbutton(bar3, text="Seguir tabuleiro sozinho", variable=self.autov, command=self.on_auto,
                       **chk).pack(side="left", padx=2)

        self.status = tk.StringVar(value="aguardando o tabuleiro..." if not self.tpl
                                   else "calibração carregada, marca 'Ativo'")
        tk.Label(root, textvariable=self.status, bg="#1b1b1b", fg="#8a8a8a").pack(anchor="w", padx=10, pady=(6, 0))
        self.big = tk.StringVar(value="—")
        tk.Label(root, textvariable=self.big, bg="#1b1b1b", fg="#e8a317",
                 font=("Segoe UI", 22, "bold")).pack(pady=4)
        self.cv = tk.Canvas(root, width=8 * kSqPx, height=8 * kSqPx, highlightthickness=0, bg="#1b1b1b")
        self.cv.pack(padx=8)
        self.txt = tk.StringVar()
        tk.Label(root, textvariable=self.txt, bg="#1b1b1b", fg="#cfcfcf", justify="left",
                 font=("Consolas", 9), wraplength=8 * kSqPx).pack(anchor="w", padx=10, pady=8)
        self.draw(chess.Board().piece_map())
        self.poll()

    # ---- config ----
    def load(self):
        try:
            with open(kCfgFile) as f:
                d = json.load(f)
            self.region = d.get("region")
            self.white_bottom = d.get("white_bottom", True)
            t = {k: np.array(v, dtype=np.float32) for k, v in (d.get("tpl") or {}).items()}
            if t and all(v.shape == (kInner, kInner) for v in t.values()):
                self.tpl = t  # calibração antiga (outro tamanho) é descartada
        except Exception:
            pass

    def save(self):
        d = {"region": self.region, "white_bottom": self.white_bottom,
             "tpl": {k: v.tolist() for k, v in (self.tpl or {}).items()}}
        with open(kCfgFile, "w") as f:
            json.dump(d, f)

    # ---- ações ----
    def clear_overlay(self):
        self.last_hint = None
        self.overlay.hide()

    def after_flash(self):
        if self.last_hint and self.ovv.get() and self.region:
            self.overlay.show(self.region, self.white_bottom, self.last_hint[1])
        else:
            self.overlay.hide()

    def flash(self):
        if self.region:
            self.overlay.grid(self.region)
            self.root.after(2500, self.after_flash)

    def on_act(self):
        self.on = self.act.get()
        self.reset = True
        self.clear_overlay()

    def on_auto(self):
        self.auto = self.autov.get()

    def on_ov(self):
        if not self.ovv.get():
            self.overlay.hide()
        elif self.last_hint and self.region:
            self.overlay.show(self.region, self.white_bottom, self.last_hint[1])

    def on_side(self, _=None):
        self.white_bottom = self.side.current() == 0
        self.reset = True
        self.clear_overlay()
        self.save()
        self.draw(chess.Board().piece_map())

    def auto_pick(self):
        self.clear_overlay()
        self.root.update_idletasks()
        self.update_self_rect()
        with mss.mss() as s:
            det = detect_board(s, self.self_rect)
        if not det:
            self.status.set("não achei o tabuleiro (tema com textura? usa Manual)")
            return
        self.region = det
        self.reset = True
        self.save()
        self.status.set(f"tabuleiro achado: {det['width']}px em ({det['left']}, {det['top']})"
                        + ("" if self.tpl else ", agora calibra"))
        self.flash()

    def pick(self):
        self.clear_overlay()
        with mss.mss() as s:
            mon = s.monitors[1]
            shot = s.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        top.geometry(f"{img.width}x{img.height}+{mon['left']}+{mon['top']}")
        top.attributes("-topmost", True)
        cv = tk.Canvas(top, width=img.width, height=img.height, highlightthickness=0, cursor="cross")
        cv.pack()
        photo = ImageTk.PhotoImage(img)
        cv.image = photo
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv.create_text(img.width // 2, 40, fill="#00ff88", font=("Segoe UI", 18, "bold"),
                       text="arrasta do canto de cima-esquerda ao de baixo-direita das 64 casas (Esc cancela)")
        st = {}

        def down(e):
            st["x"], st["y"] = e.x, e.y
            st["id"] = cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#00ff88", width=2)

        def move(e):
            if "id" in st:
                cv.coords(st["id"], st["x"], st["y"], e.x, e.y)

        def up(e):
            if "id" not in st:
                return
            x0, x1 = sorted((st["x"], e.x))
            y0, y1 = sorted((st["y"], e.y))
            side = min(x1 - x0, y1 - y0)
            if side < 160:  # clique solto: não fecha, arrasta de novo
                cv.delete(st.pop("id"))
                return
            top.destroy()
            self.region = {"left": mon["left"] + x0, "top": mon["top"] + y0, "width": side, "height": side}
            self.save()
            self.reset = True
            self.status.set("tabuleiro selecionado" + ("" if self.tpl else ", agora calibra"))
            self.flash()

        cv.bind("<ButtonPress-1>", down)
        cv.bind("<B1-Motion>", move)
        cv.bind("<ButtonRelease-1>", up)
        top.bind("<Escape>", lambda e: top.destroy())
        top.focus_force()

    def calibrate(self):
        if not self.region:
            messagebox.showinfo("Calibrar", "ainda sem tabuleiro: clica em 'Achar tabuleiro (auto)'")
            return
        self.clear_overlay()
        with mss.mss() as s:
            img = grab(s, self.region)
        score, tpl, b = calibrate_templates(img, self.white_bottom)
        self.tpl = tpl
        self.reset = True
        self.save()
        what = chess.Board().variation_san(b.move_stack) if b.move_stack else "posição inicial"
        self.status.set(f"calibrado ({what}), salvo. marca 'Ativo'")

    # ---- UI ----
    def update_self_rect(self):
        r = self.root
        self.self_rect = (r.winfo_rootx(), r.winfo_rooty(), r.winfo_width(), r.winfo_height())

    def poll(self):
        self.update_self_rect()
        try:
            while True:
                kind, *a = self.q.get_nowait()
                if kind == "status":
                    self.status.set(a[0])
                elif kind == "region":  # worker achou/reajustou sozinho
                    self.region = a[0]
                    self.save()
                    self.clear_overlay()
                    self.status.set(f"tabuleiro achado: {a[0]['width']}px" + ("" if self.tpl else ", agora calibra"))
                    self.flash()
                elif kind == "seen":
                    s = a[0]
                    self.draw({sq: chess.Piece.from_symbol(p) for sq, p in s.items()})
                    if self.last_hint and self.ovv.get() and self.region:
                        if s == self.last_hint[0]:  # tabuleiro igual ao do hint: marcação vale
                            self.overlay.show(self.region, self.white_bottom, self.last_hint[1])
                        else:                        # tu jogou / posição mudou: some
                            self.overlay.hide()
                elif kind == "wait":
                    self.status.set(a[0])
                    self.big.set("…")
                    self.txt.set("")
                    self.clear_overlay()
                elif kind == "hint":
                    board, lines = a
                    best = lines[0]
                    mv = chess.Move.from_uci(best[0])
                    self.status.set("tua vez")
                    self.big.set(f"{best[1]}   ({best[2]})")
                    self.txt.set("\n".join(f"{i + 1}. {l[1]:7} {l[2]:>6}  {l[3]}" for i, l in enumerate(lines)))
                    self.draw(board.piece_map(), mv)
                    self.last_hint = (placement(board), mv)
                    if self.ovv.get() and self.region:
                        if not self.overlay.show(self.region, self.white_bottom, mv):
                            self.status.set("tua vez (overlay sem seta: Windows sem captura exclusiva)")
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def draw(self, pieces, mv=None):
        c, wb = self.cv, self.white_bottom
        c.delete("all")
        for sq in chess.SQUARES:
            r, col = sq_to_cell(sq, wb)
            x, y = col * kSqPx, r * kSqPx
            fill = kLightSq if (r + col) % 2 == 0 else kDarkSq
            if mv and sq in (mv.from_square, mv.to_square):
                fill = "#aec85a"
            c.create_rectangle(x, y, x + kSqPx, y + kSqPx, fill=fill, width=0)
            p = pieces.get(sq)
            if p:
                g = chess.Piece(p.piece_type, chess.BLACK).unicode_symbol()
                cx, cy = x + kSqPx / 2, y + kSqPx / 2
                f = ("Segoe UI Symbol", 28)
                if p.color:
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        c.create_text(cx + dx, cy + dy, text=g, font=f, fill="#111")
                c.create_text(cx, cy, text=g, font=f, fill="#fafafa" if p.color else "#111")
        if mv:
            def ctr(s):
                r, col = sq_to_cell(s, wb)
                return col * kSqPx + kSqPx / 2, r * kSqPx + kSqPx / 2
            c.create_line(*ctr(mv.from_square), *ctr(mv.to_square), arrow=tk.LAST, width=7,
                          fill="#e8a317", arrowshape=(18, 20, 8))


def main():
    root = tk.Tk()
    app = App(root)
    threading.Thread(target=worker, args=(app,), daemon=True).start()
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(app, "alive", False), root.destroy()))
    root.mainloop()


try:
    main()
except Exception:
    traceback.print_exc()
    input("erro acima, enter pra fechar")
