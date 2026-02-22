# GeoConflictModeler UCI Engine (Permissive)
# ------------------------------------------------------------
# A self-contained UCI chess engine written from scratch in Python.
# - No Stockfish
# - No python-chess dependency
# - Intended for server/online deployment with minimal licensing friction.
#
# Features:
# - NegaMax + AlphaBeta (PVS)
# - Quiescence (captures + promotions)
# - Transposition table (Zobrist)
# - Move ordering: hash move, captures (MVV-LVA), killers, history
# - Null-move pruning (conservative) + late move reductions (LMR)
# - Iterative deepening + aspiration windows
#
# License: MIT
# Copyright (c) 2026
# ------------------------------------------------------------

from __future__ import annotations

import sys
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# -----------------------------
# Board representation
# -----------------------------
# Squares: 0..63 with a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
# Pieces: 0 empty; positive = White, negative = Black
#   1=P, 2=N, 3=B, 4=R, 5=Q, 6=K

P, N, B, R, Q, K = 1, 2, 3, 4, 5, 6

FILES = "abcdefgh"
RANKS = "12345678"

MATE = 30000
INF = 10**9

# Material (centipawns)
MG_VAL = {P: 82, N: 337, B: 365, R: 477, Q: 1025, K: 0}
EG_VAL = {P: 94, N: 281, B: 297, R: 512, Q: 936, K: 0}

PHASE_INC = {P: 0, N: 1, B: 1, R: 2, Q: 4, K: 0}
TOTAL_PHASE = 24

# PSTs: from White POV. Black uses mirror(square) = square ^ 56.
# (Same tables as the previous python-chess-based scaffold.)
MG_PST: Dict[int, List[int]] = {
    P: [
         0,  0,  0,  0,  0,  0,  0,  0,
        98,134, 61, 95, 68,126, 34,-11,
        -6,  7, 26, 31, 65, 56, 25,-20,
       -14, 13,  6, 21, 23, 12, 17,-23,
       -27, -2, -5, 12, 17,  6, 10,-25,
       -26, -4, -4,-10,  3,  3, 33,-12,
       -35, -1,-20,-23,-15, 24, 38,-22,
         0,  0,  0,  0,  0,  0,  0,  0,
    ],
    N: [
       -167,-89,-34,-49, 61,-97,-15,-107,
        -73,-41, 72, 36, 23, 62,  7, -17,
        -47, 60, 37, 65, 84,129, 73,  44,
         -9, 17, 19, 53, 37, 69, 18,  22,
        -13,  4, 16, 13, 28, 19, 21,  -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29,-53,-12, -3, -1, 18,-14, -19,
       -105,-21,-58,-33,-17,-28,-19, -23,
    ],
    B: [
        -29,  4,-82,-37,-25,-42,  7, -8,
        -26, 16,-18,-13, 30, 59, 18,-47,
        -16, 37, 43, 40, 35, 50, 37, -2,
         -4,  5, 19, 50, 37, 37,  7, -2,
         -6, 13, 13, 26, 34, 12, 10,  4,
          0, 15, 15, 15, 14, 27, 18, 10,
          4, 15, 16,  0,  7, 21, 33,  1,
        -33, -3,-14,-21,-13,-12,-39,-21,
    ],
    R: [
         32, 42, 32, 51, 63,  9, 31, 43,
         27, 32, 58, 62, 80, 67, 26, 44,
         -5, 19, 26, 36, 17, 45, 61, 16,
        -24,-11,  7, 26, 24, 35, -8,-20,
        -36,-26,-12, -1,  9, -7,  6,-23,
        -45,-25,-16,-17,  3,  0, -5,-33,
        -44,-16,-20, -9, -1, 11, -6,-71,
        -19,-13,  1, 17, 16,  7,-37,-26,
    ],
    Q: [
        -28,  0, 29, 12, 59, 44, 43, 45,
        -24,-39, -5,  1,-16, 57, 28, 54,
        -13,-17,  7,  8, 29, 56, 47, 57,
        -27,-27,-16,-16, -1, 17, -2,  1,
         -9,-26, -9,-10, -2, -4,  3, -3,
        -14,  2,-11, -2, -5,  2, 14,  5,
        -35, -8, 11,  2,  8, 15, -3,  1,
         -1,-18, -9, 10,-15,-25,-31,-50,
    ],
    K: [
        -65, 23, 16,-15,-56,-34,  2, 13,
         29, -1,-20, -7, -8, -4,-38,-29,
         -9, 24,  2,-16,-20,  6, 22,-22,
        -17,-20,-12,-27,-30,-25,-14,-36,
        -49, -1,-27,-39,-46,-44,-33,-51,
        -14,-14,-22,-46,-44,-30,-15,-27,
          1,  7, -8,-64,-43,-16,  9,  8,
        -15, 36, 12,-54,  8,-28, 24, 14,
    ],
}

EG_PST: Dict[int, List[int]] = {
    P: [
         0,  0,  0,  0,  0,  0,  0,  0,
        178,173,158,134,147,132,165,187,
         94,100, 85, 67, 56, 53, 82, 84,
         32, 24, 13,  5, -2,  4, 17, 17,
         13,  9, -3, -7, -7, -8,  3, -1,
          4,  7, -6,  1,  0, -5, -1, -8,
         13,  8,  8, 10, 13,  0,  2, -7,
          0,  0,  0,  0,  0,  0,  0,  0,
    ],
    N: [
        -58,-38,-13,-28,-31,-27,-63,-99,
        -25, -8,-25, -2, -9,-25,-24,-52,
        -24,-20, 10,  9, -1, -9,-19,-41,
        -17,  3, 22, 22, 22, 11,  8,-18,
        -18, -6, 16, 25, 16, 17,  4,-18,
        -23, -3, -1, 15, 10, -3,-20,-22,
        -42,-20,-10, -5, -2,-20,-23,-44,
        -29,-51,-23,-15,-22,-18,-50,-64,
    ],
    B: [
        -14,-21,-11, -8, -7, -9,-17,-24,
         -8, -4,  7,-12, -3,-13, -4,-14,
          2, -8,  0, -1, -2,  6,  0,  4,
         -3,  9, 12,  9, 14, 10,  3,  2,
         -6,  3, 13, 19,  7, 10, -3, -9,
        -12, -3,  8, 10, 13,  3, -7,-15,
        -14,-18, -7, -1,  4, -9,-15,-27,
        -23, -9,-23, -5, -9,-16, -5,-17,
    ],
    R: [
        13, 10, 18, 15, 12, 12,  8,  5,
        11, 13, 13, 11, -3,  3,  8,  3,
         7,  7,  7,  5,  4, -3, -5, -3,
         4,  3, 13,  1,  2,  1, -1,  2,
         3,  5,  8,  4, -5, -6, -8,-11,
        -4,  0, -5, -1, -7,-12, -8,-16,
        -6, -6,  0,  2, -9, -9,-11, -3,
        -9,  2,  3, -1, -5,-13,  4,-20,
    ],
    Q: [
        -9, 22, 22, 27, 27, 19, 10, 20,
       -17, 20, 32, 41, 58, 25, 30,  0,
       -20,  6,  9, 49, 47, 35, 19,  9,
         3, 22, 24, 45, 57, 40, 57, 36,
       -18, 28, 19, 47, 31, 34, 39, 23,
       -16,-27, 15,  6,  9, 17, 10,  5,
       -22,-23,-30,-16,-16,-23,-36,-32,
       -33,-28,-22,-43, -5,-32,-20,-41,
    ],
    K: [
        -74,-35,-18,-18,-11, 15,  4,-17,
        -12, 17, 14, 17, 17, 38, 23, 11,
         10, 17, 23, 15, 20, 45, 44, 13,
         -8, 22, 24, 27, 26, 33, 26,  3,
        -18, -4, 21, 24, 27, 23,  9,-11,
        -19, -3, 11, 21, 23, 16,  7, -9,
        -27,-11,  4, 13, 14,  4, -5,-17,
        -53,-34,-21,-11,-28,-14,-24,-43,
    ],
}

PIECE_VALUE = {P: 100, N: 320, B: 330, R: 500, Q: 900, K: 20000}

KNIGHT_OFFS = [17, 15, 10, 6, -17, -15, -10, -6]
KING_OFFS = [1, -1, 8, -8, 9, 7, -7, -9]

BISH_DIRS = [9, 7, -7, -9]
ROOK_DIRS = [8, -8, 1, -1]


def mirror_sq(sq: int) -> int:
    return sq ^ 56


def sq_file(sq: int) -> int:
    return sq & 7


def sq_rank(sq: int) -> int:
    return sq >> 3


def in_bounds(sq: int) -> bool:
    return 0 <= sq < 64


def uci_sq(sq: int) -> str:
    return FILES[sq_file(sq)] + RANKS[sq_rank(sq)]


def parse_uci_sq(s: str) -> int:
    f = FILES.index(s[0])
    r = RANKS.index(s[1])
    return r * 8 + f


@dataclass
class Move:
    frm: int
    to: int
    promo: int = 0  # 0 none, else N/B/R/Q encoded as N,B,R,Q

    def uci(self) -> str:
        if self.promo:
            promo_ch = {N: 'n', B: 'b', R: 'r', Q: 'q'}[self.promo]
            return f"{uci_sq(self.frm)}{uci_sq(self.to)}{promo_ch}"
        return f"{uci_sq(self.frm)}{uci_sq(self.to)}"


class Zobrist:
    def __init__(self, seed: int = 0xC0FFEE):
        rng = random.Random(seed)
        # 12 piece indices * 64
        self.piece_sq = [rng.getrandbits(64) for _ in range(12 * 64)]
        self.side = rng.getrandbits(64)
        self.castle = [rng.getrandbits(64) for _ in range(16)]
        self.ep_file = [rng.getrandbits(64) for _ in range(8)]

    @staticmethod
    def piece_index(piece_code: int) -> int:
        # piece_code: +1..+6 white, -1..-6 black
        pt = abs(piece_code)
        is_black = 1 if piece_code < 0 else 0
        return (pt - 1) * 2 + is_black

    def piece_key(self, piece_code: int, sq: int) -> int:
        pi = self.piece_index(piece_code)
        return self.piece_sq[pi * 64 + sq]


class Board:
    def __init__(self):
        self.pieces = [0] * 64
        self.turn = 1  # 1 white, -1 black
        self.castle = 0  # WK=1,WQ=2,BK=4,BQ=8
        self.ep = -1
        self.halfmove = 0
        self.fullmove = 1
        self.king_w = 4
        self.king_b = 60

        self.z = Zobrist()
        self.key = 0
        self.key_history: List[int] = []

    def set_startpos(self) -> None:
        self.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    def _recompute_key(self) -> int:
        k = 0
        for sq in range(64):
            pc = self.pieces[sq]
            if pc:
                k ^= self.z.piece_key(pc, sq)
        if self.turn == -1:
            k ^= self.z.side
        k ^= self.z.castle[self.castle & 15]
        if self.ep != -1:
            k ^= self.z.ep_file[sq_file(self.ep)]
        return k

    def set_fen(self, fen: str) -> None:
        parts = fen.strip().split()
        if len(parts) < 4:
            raise ValueError("bad FEN")
        board_part, stm, cast, ep = parts[0], parts[1], parts[2], parts[3]
        half = int(parts[4]) if len(parts) >= 5 else 0
        full = int(parts[5]) if len(parts) >= 6 else 1

        self.pieces = [0] * 64
        self.turn = 1 if stm == 'w' else -1
        self.castle = 0
        if 'K' in cast:
            self.castle |= 1
        if 'Q' in cast:
            self.castle |= 2
        if 'k' in cast:
            self.castle |= 4
        if 'q' in cast:
            self.castle |= 8
        self.ep = -1 if ep == '-' else parse_uci_sq(ep)
        self.halfmove = half
        self.fullmove = full

        ranks = board_part.split('/')
        if len(ranks) != 8:
            raise ValueError("bad board")
        for r in range(8):
            file_ = 0
            row = ranks[r]
            for ch in row:
                if ch.isdigit():
                    file_ += int(ch)
                    continue
                sq = (7 - r) * 8 + file_
                file_ += 1
                if ch == 'P': pc = P
                elif ch == 'N': pc = N
                elif ch == 'B': pc = B
                elif ch == 'R': pc = R
                elif ch == 'Q': pc = Q
                elif ch == 'K': pc = K
                elif ch == 'p': pc = -P
                elif ch == 'n': pc = -N
                elif ch == 'b': pc = -B
                elif ch == 'r': pc = -R
                elif ch == 'q': pc = -Q
                elif ch == 'k': pc = -K
                else:
                    raise ValueError("bad piece")
                self.pieces[sq] = pc
                if pc == K:
                    self.king_w = sq
                elif pc == -K:
                    self.king_b = sq

        self.key = self._recompute_key()
        self.key_history = [self.key]

    def fen(self) -> str:
        def piece_char(pc: int) -> str:
            m = {P:'P',N:'N',B:'B',R:'R',Q:'Q',K:'K'}
            ch = m[abs(pc)]
            return ch.lower() if pc < 0 else ch

        rows = []
        for r in range(7, -1, -1):
            empt = 0
            s = ""
            for f in range(8):
                sq = r * 8 + f
                pc = self.pieces[sq]
                if pc == 0:
                    empt += 1
                else:
                    if empt:
                        s += str(empt)
                        empt = 0
                    s += piece_char(pc)
            if empt:
                s += str(empt)
            rows.append(s)

        stm = 'w' if self.turn == 1 else 'b'
        cast = ""
        cast += 'K' if (self.castle & 1) else ''
        cast += 'Q' if (self.castle & 2) else ''
        cast += 'k' if (self.castle & 4) else ''
        cast += 'q' if (self.castle & 8) else ''
        if not cast:
            cast = '-'
        ep = '-' if self.ep == -1 else uci_sq(self.ep)
        return f"{'/'.join(rows)} {stm} {cast} {ep} {self.halfmove} {self.fullmove}"

    def is_in_check(self, color: int) -> bool:
        ksq = self.king_w if color == 1 else self.king_b
        return self.is_square_attacked(ksq, -color)

    def is_square_attacked(self, sq: int, by_color: int) -> bool:
        # Pawn attacks
        if by_color == 1:
            # white pawns attack from sq-7, sq-9
            for d in (-7, -9):
                frm = sq + d
                if in_bounds(frm):
                    if self.pieces[frm] == P and abs(sq_file(frm) - sq_file(sq)) == 1:
                        return True
        else:
            # black pawns attack from sq+7, sq+9
            for d in (7, 9):
                frm = sq + d
                if in_bounds(frm):
                    if self.pieces[frm] == -P and abs(sq_file(frm) - sq_file(sq)) == 1:
                        return True

        # Knights
        for d in KNIGHT_OFFS:
            frm = sq + d
            if not in_bounds(frm):
                continue
            # file wrap protection
            if abs(sq_file(frm) - sq_file(sq)) not in (1, 2):
                continue
            pc = self.pieces[frm]
            if pc == (N if by_color == 1 else -N):
                return True

        # Bishops / Queens
        for d in BISH_DIRS:
            cur = sq + d
            while in_bounds(cur) and abs(sq_file(cur) - sq_file(cur - d)) == 1:
                pc = self.pieces[cur]
                if pc:
                    if pc == (B if by_color == 1 else -B) or pc == (Q if by_color == 1 else -Q):
                        return True
                    break
                cur += d

        # Rooks / Queens
        for d in ROOK_DIRS:
            cur = sq + d
            while in_bounds(cur) and (d in (8, -8) or sq_file(cur) - sq_file(cur - d) in (1, -1)):
                if d in (1, -1) and abs(sq_file(cur) - sq_file(cur - d)) != 1:
                    break
                pc = self.pieces[cur]
                if pc:
                    if pc == (R if by_color == 1 else -R) or pc == (Q if by_color == 1 else -Q):
                        return True
                    break
                cur += d

        # King
        for d in KING_OFFS:
            frm = sq + d
            if not in_bounds(frm):
                continue
            if abs(sq_file(frm) - sq_file(sq)) > 1:
                continue
            pc = self.pieces[frm]
            if pc == (K if by_color == 1 else -K):
                return True

        return False

    def _castle_rights_code(self) -> int:
        return self.castle & 15

    def _xor_castle(self, old: int, new: int) -> None:
        self.key ^= self.z.castle[old & 15]
        self.key ^= self.z.castle[new & 15]

    def _xor_ep(self, old_ep: int, new_ep: int) -> None:
        if old_ep != -1:
            self.key ^= self.z.ep_file[sq_file(old_ep)]
        if new_ep != -1:
            self.key ^= self.z.ep_file[sq_file(new_ep)]

    def _set_piece(self, sq: int, new_pc: int, changes: List[Tuple[int, int]]) -> None:
        old_pc = self.pieces[sq]
        if old_pc == new_pc:
            return
        if old_pc:
            self.key ^= self.z.piece_key(old_pc, sq)
        if new_pc:
            self.key ^= self.z.piece_key(new_pc, sq)
        changes.append((sq, old_pc))
        self.pieces[sq] = new_pc

    def push(self, mv: Move) -> Tuple[List[Tuple[int, int]], int, int, int, int, int, int, int]:
        """Make move and return undo tuple."""
        changes: List[Tuple[int, int]] = []
        prev_castle = self.castle
        prev_ep = self.ep
        prev_half = self.halfmove
        prev_full = self.fullmove
        prev_key = self.key
        prev_kw = self.king_w
        prev_kb = self.king_b

        # Clear EP
        self._xor_ep(prev_ep, -1)
        self.ep = -1

        pc = self.pieces[mv.frm]
        captured = self.pieces[mv.to]
        captured_sq = mv.to

        # Halfmove clock
        if abs(pc) == P or captured != 0:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # Castling rights changes due to moving/capturing rook/king
        new_castle = self.castle
        if pc == K:
            new_castle &= ~3
        elif pc == -K:
            new_castle &= ~12
        elif pc == R and mv.frm == 7:
            new_castle &= ~1
        elif pc == R and mv.frm == 0:
            new_castle &= ~2
        elif pc == -R and mv.frm == 63:
            new_castle &= ~4
        elif pc == -R and mv.frm == 56:
            new_castle &= ~8

        # Capturing rooks on corners
        if captured == R and mv.to == 7:
            new_castle &= ~1
        elif captured == R and mv.to == 0:
            new_castle &= ~2
        elif captured == -R and mv.to == 63:
            new_castle &= ~4
        elif captured == -R and mv.to == 56:
            new_castle &= ~8

        if new_castle != self.castle:
            self._xor_castle(self.castle, new_castle)
            self.castle = new_castle

        # Handle special moves
        is_ep = (abs(pc) == P and mv.to == prev_ep and captured == 0 and abs(sq_file(mv.to) - sq_file(mv.frm)) == 1)
        is_castle = (abs(pc) == K and abs(mv.to - mv.frm) == 2)

        if is_ep:
            # captured pawn is behind to-square
            cap_sq = mv.to - 8 if pc > 0 else mv.to + 8
            captured = self.pieces[cap_sq]
            captured_sq = cap_sq
            self._set_piece(cap_sq, 0, changes)

        # Move piece (with promotion)
        self._set_piece(mv.frm, 0, changes)

        moved_pc = pc
        if mv.promo and abs(pc) == P:
            moved_pc = mv.promo if pc > 0 else -mv.promo
        self._set_piece(mv.to, moved_pc, changes)

        # Update king location
        if pc == K:
            self.king_w = mv.to
        elif pc == -K:
            self.king_b = mv.to

        # Castling rook move
        if is_castle:
            if pc == K and mv.to == 6:  # e1g1
                self._set_piece(7, 0, changes)
                self._set_piece(5, R, changes)
            elif pc == K and mv.to == 2:  # e1c1
                self._set_piece(0, 0, changes)
                self._set_piece(3, R, changes)
            elif pc == -K and mv.to == 62:  # e8g8
                self._set_piece(63, 0, changes)
                self._set_piece(61, -R, changes)
            elif pc == -K and mv.to == 58:  # e8c8
                self._set_piece(56, 0, changes)
                self._set_piece(59, -R, changes)

        # Set new EP if pawn double
        if abs(pc) == P and abs(mv.to - mv.frm) == 16:
            new_ep = (mv.frm + mv.to) // 2
            self._xor_ep(-1, new_ep)
            self.ep = new_ep

        # Side to move
        self.turn = -self.turn
        self.key ^= self.z.side

        if self.turn == 1:
            self.fullmove += 1

        self.key_history.append(self.key)

        return changes, captured, captured_sq, prev_castle, prev_ep, prev_half, prev_full, prev_key, prev_kw, prev_kb

    def pop(self, undo) -> None:
        changes, captured, captured_sq, prev_castle, prev_ep, prev_half, prev_full, prev_key, prev_kw, prev_kb = undo
        # revert key/state quickly
        self.key = prev_key
        self.castle = prev_castle
        self.ep = prev_ep
        self.halfmove = prev_half
        self.fullmove = prev_full
        self.king_w = prev_kw
        self.king_b = prev_kb
        # side
        self.turn = -self.turn

        # revert pieces
        for sq, old_pc in reversed(changes):
            self.pieces[sq] = old_pc

        if self.key_history:
            self.key_history.pop()

    def legal_moves(self) -> List[Move]:
        moves = self.pseudo_moves()
        out: List[Move] = []
        for mv in moves:
            undo = self.push(mv)
            if not self.is_in_check(-self.turn):
                out.append(mv)
            self.pop(undo)
        return out

    def pseudo_moves(self) -> List[Move]:
        out: List[Move] = []
        us = self.turn
        for sq in range(64):
            pc = self.pieces[sq]
            if pc == 0 or (pc > 0) != (us > 0):
                continue
            pt = abs(pc)

            if pt == P:
                dir_ = 8 if us == 1 else -8
                start_rank = 1 if us == 1 else 6
                promo_rank = 7 if us == 1 else 0

                fwd = sq + dir_
                if in_bounds(fwd) and self.pieces[fwd] == 0:
                    if sq_rank(fwd) == promo_rank:
                        for pr in (Q, R, B, N):
                            out.append(Move(sq, fwd, pr))
                    else:
                        out.append(Move(sq, fwd))
                        if sq_rank(sq) == start_rank:
                            fwd2 = sq + 2 * dir_
                            if self.pieces[fwd2] == 0:
                                out.append(Move(sq, fwd2))

                for dc in (-1, 1):
                    cap = sq + dir_ + dc
                    if not in_bounds(cap):
                        continue
                    if abs(sq_file(cap) - sq_file(sq)) != 1:
                        continue
                    tgt = self.pieces[cap]
                    if tgt != 0 and (tgt > 0) != (us > 0):
                        if sq_rank(cap) == promo_rank:
                            for pr in (Q, R, B, N):
                                out.append(Move(sq, cap, pr))
                        else:
                            out.append(Move(sq, cap))

                # En passant
                if self.ep != -1:
                    if abs(sq_file(self.ep) - sq_file(sq)) == 1 and (self.ep == sq + dir_ - 1 or self.ep == sq + dir_ + 1):
                        out.append(Move(sq, self.ep))

            elif pt == N:
                for d in KNIGHT_OFFS:
                    to = sq + d
                    if not in_bounds(to):
                        continue
                    if abs(sq_file(to) - sq_file(sq)) not in (1, 2):
                        continue
                    tgt = self.pieces[to]
                    if tgt == 0 or (tgt > 0) != (us > 0):
                        out.append(Move(sq, to))

            elif pt == B or pt == R or pt == Q:
                dirs = BISH_DIRS if pt == B else ROOK_DIRS if pt == R else (BISH_DIRS + ROOK_DIRS)
                for d in dirs:
                    cur = sq + d
                    while in_bounds(cur):
                        # file wrap constraints
                        if d in (1, -1, 9, -9, 7, -7):
                            if abs(sq_file(cur) - sq_file(cur - d)) != 1:
                                break
                        tgt = self.pieces[cur]
                        if tgt == 0:
                            out.append(Move(sq, cur))
                        else:
                            if (tgt > 0) != (us > 0):
                                out.append(Move(sq, cur))
                            break
                        cur += d

            elif pt == K:
                for d in KING_OFFS:
                    to = sq + d
                    if not in_bounds(to):
                        continue
                    if abs(sq_file(to) - sq_file(sq)) > 1:
                        continue
                    tgt = self.pieces[to]
                    if tgt == 0 or (tgt > 0) != (us > 0):
                        out.append(Move(sq, to))

                # Castling
                if us == 1 and sq == 4:
                    if (self.castle & 1) and self.pieces[5] == 0 and self.pieces[6] == 0:
                        if (not self.is_square_attacked(4, -1)) and (not self.is_square_attacked(5, -1)) and (not self.is_square_attacked(6, -1)):
                            out.append(Move(4, 6))
                    if (self.castle & 2) and self.pieces[3] == 0 and self.pieces[2] == 0 and self.pieces[1] == 0:
                        if (not self.is_square_attacked(4, -1)) and (not self.is_square_attacked(3, -1)) and (not self.is_square_attacked(2, -1)):
                            out.append(Move(4, 2))
                elif us == -1 and sq == 60:
                    if (self.castle & 4) and self.pieces[61] == 0 and self.pieces[62] == 0:
                        if (not self.is_square_attacked(60, 1)) and (not self.is_square_attacked(61, 1)) and (not self.is_square_attacked(62, 1)):
                            out.append(Move(60, 62))
                    if (self.castle & 8) and self.pieces[59] == 0 and self.pieces[58] == 0 and self.pieces[57] == 0:
                        if (not self.is_square_attacked(60, 1)) and (not self.is_square_attacked(59, 1)) and (not self.is_square_attacked(58, 1)):
                            out.append(Move(60, 58))

        return out

    def is_drawish(self) -> bool:
        # 50-move rule and simple repetition (3-fold claim)
        if self.halfmove >= 100:
            return True
        # repetition count of current key in history
        k = self.key
        cnt = 0
        for kk in self.key_history:
            if kk == k:
                cnt += 1
                if cnt >= 3:
                    return True
        return False


def tapered_eval_white_pov(b: Board) -> int:
    mg = 0
    eg = 0
    phase = 0
    for sq in range(64):
        pc = b.pieces[sq]
        if not pc:
            continue
        pt = abs(pc)
        phase += PHASE_INC[pt]
        if pc > 0:
            mg += MG_VAL[pt] + MG_PST[pt][sq]
            eg += EG_VAL[pt] + EG_PST[pt][sq]
        else:
            msq = mirror_sq(sq)
            mg -= MG_VAL[pt] + MG_PST[pt][msq]
            eg -= EG_VAL[pt] + EG_PST[pt][msq]

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    return (mg * phase + eg * (TOTAL_PHASE - phase)) // TOTAL_PHASE


def eval_side_to_move(b: Board) -> int:
    score = tapered_eval_white_pov(b)
    return score if b.turn == 1 else -score


def mvv_lva(b: Board, mv: Move) -> int:
    tgt = b.pieces[mv.to]
    if tgt == 0:
        # en-passant victim
        pc = b.pieces[mv.frm]
        if abs(pc) == P and mv.to == b.ep and abs(sq_file(mv.to) - sq_file(mv.frm)) == 1:
            victim = PIECE_VALUE[P]
        else:
            return 0
    else:
        victim = PIECE_VALUE[abs(tgt)]
    aggressor = b.pieces[mv.frm]
    return 10 * victim - PIECE_VALUE[abs(aggressor)]


TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2


@dataclass
class TTEntry:
    key: int
    depth: int
    value: int
    flag: int
    move: Optional[Move]


class TranspositionTable:
    def __init__(self, size_pow2: int = 1 << 18):
        self.size = size_pow2
        self.mask = size_pow2 - 1
        self.table: List[Optional[TTEntry]] = [None] * size_pow2

    def probe(self, key: int) -> Optional[TTEntry]:
        e = self.table[key & self.mask]
        if e is not None and e.key == key:
            return e
        return None

    def store(self, e: TTEntry) -> None:
        idx = e.key & self.mask
        cur = self.table[idx]
        if cur is None or cur.key != e.key or e.depth >= cur.depth:
            self.table[idx] = e


class SearchTimeout(Exception):
    pass


class Engine:
    def __init__(self):
        self.tt = TranspositionTable(1 << 18)
        self.nodes = 0
        self.start_time = 0.0
        self.time_limit = 0.0
        self.killers: List[List[Optional[Move]]] = [[None, None] for _ in range(256)]
        self.history = [[0 for _ in range(64)] for _ in range(64)]

    def set_hash_mb(self, mb: int) -> None:
        target = max(1 << 15, min(1 << 21, (mb * (1 << 20)) // 64))
        size = 1
        while size < target:
            size <<= 1
        self.tt = TranspositionTable(size)

    def _check_time(self) -> None:
        if self.time_limit > 0 and (time.time() - self.start_time) >= self.time_limit:
            raise SearchTimeout

    def quiescence(self, b: Board, alpha: int, beta: int) -> int:
        self._check_time()
        self.nodes += 1

        stand_pat = eval_side_to_move(b)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        moves = []
        for mv in b.pseudo_moves():
            if b.pieces[mv.to] != 0 or mv.promo:
                moves.append(mv)
            else:
                # en-passant capture counts
                pc = b.pieces[mv.frm]
                if abs(pc) == P and mv.to == b.ep and abs(sq_file(mv.to) - sq_file(mv.frm)) == 1:
                    moves.append(mv)

        moves.sort(key=lambda m: mvv_lva(b, m), reverse=True)

        for mv in moves:
            undo = b.push(mv)
            if b.is_in_check(-b.turn):
                b.pop(undo)
                continue
            score = -self.quiescence(b, -beta, -alpha)
            b.pop(undo)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def negamax(self, b: Board, depth: int, alpha: int, beta: int, ply: int, null_ok: bool = True) -> int:
        self._check_time()

        if b.is_in_check(b.turn):
            in_check = True
        else:
            in_check = False

        # terminal
        legal = None
        if depth <= 0:
            return self.quiescence(b, alpha, beta)

        if b.is_drawish():
            return 0

        key = b.key
        tt_e = self.tt.probe(key)
        if tt_e is not None and tt_e.depth >= depth:
            if tt_e.flag == TT_EXACT:
                return tt_e.value
            if tt_e.flag == TT_LOWER:
                alpha = max(alpha, tt_e.value)
            elif tt_e.flag == TT_UPPER:
                beta = min(beta, tt_e.value)
            if alpha >= beta:
                return tt_e.value

        # Null-move pruning (conservative)
        if null_ok and depth >= 3 and (not in_check):
            # avoid null in very low material (zugzwang-ish): require some non-pawn material
            non_pawn = 0
            for pc in b.pieces:
                if pc and abs(pc) in (N, B, R, Q):
                    non_pawn += 1
                    break
            if non_pawn:
                # make null move: just flip turn, clear ep
                prev_key = b.key
                prev_turn = b.turn
                prev_ep = b.ep
                prev_hist_len = len(b.key_history)

                # XOR ep out
                if b.ep != -1:
                    b.key ^= b.z.ep_file[sq_file(b.ep)]
                b.ep = -1

                b.turn = -b.turn
                b.key ^= b.z.side
                b.key_history.append(b.key)

                Rn = 2
                try:
                    score = -self.negamax(b, depth - 1 - Rn, -beta, -beta + 1, ply + 1, null_ok=False)
                finally:
                    # undo null
                    b.key_history = b.key_history[:prev_hist_len]
                    b.turn = prev_turn
                    b.ep = prev_ep
                    b.key = prev_key

                if score >= beta:
                    return beta

        self.nodes += 1

        best_val = -INF
        best_move: Optional[Move] = None
        alpha0 = alpha

        # Generate legal moves (expensive but safe)
        moves = b.legal_moves()
        if not moves:
            # checkmate or stalemate
            if in_check:
                return -MATE + ply
            return 0

        # ordering
        hash_move = tt_e.move if tt_e is not None else None

        def ord_key(mv: Move) -> Tuple[int, int, int]:
            if hash_move is not None and mv.frm == hash_move.frm and mv.to == hash_move.to and mv.promo == hash_move.promo:
                return (10_000_000, 0, 0)
            tgt = b.pieces[mv.to]
            if tgt != 0 or mv.promo:
                return (9_000_000 + mvv_lva(b, mv), 0, 0)
            k0, k1 = self.killers[ply]
            if k0 is not None and mv.frm == k0.frm and mv.to == k0.to and mv.promo == k0.promo:
                return (8_000_000, 0, 0)
            if k1 is not None and mv.frm == k1.frm and mv.to == k1.to and mv.promo == k1.promo:
                return (7_500_000, 0, 0)
            return (self.history[mv.frm][mv.to], 0, 0)

        moves.sort(key=ord_key, reverse=True)

        # PVS
        first = True
        for idx, mv in enumerate(moves):
            # LMR for late quiet moves
            reduction = 0
            tgt = b.pieces[mv.to]
            is_quiet = (tgt == 0 and mv.promo == 0)
            if depth >= 3 and idx >= 4 and is_quiet and (not in_check):
                reduction = 1

            undo = b.push(mv)
            if b.is_in_check(-b.turn):
                b.pop(undo)
                continue

            ext = 1 if b.is_in_check(b.turn) else 0
            d2 = depth - 1 + ext - reduction

            if first:
                val = -self.negamax(b, d2, -beta, -alpha, ply + 1)
                first = False
            else:
                # zero window search
                val = -self.negamax(b, d2, -(alpha + 1), -alpha, ply + 1)
                if val > alpha and val < beta:
                    val = -self.negamax(b, d2, -beta, -alpha, ply + 1)

            b.pop(undo)

            if val > best_val:
                best_val = val
                best_move = mv

            if val > alpha:
                alpha = val

            if alpha >= beta:
                # update killers/history for quiet
                if is_quiet:
                    k0, k1 = self.killers[ply]
                    if k0 is None or (k0.frm != mv.frm or k0.to != mv.to or k0.promo != mv.promo):
                        self.killers[ply][1] = k0
                        self.killers[ply][0] = mv
                    self.history[mv.frm][mv.to] += depth * depth
                break

        flag = TT_EXACT
        if best_val <= alpha0:
            flag = TT_UPPER
        elif best_val >= beta:
            flag = TT_LOWER

        self.tt.store(TTEntry(key=key, depth=depth, value=best_val, flag=flag, move=best_move))
        return best_val

    def search(self, b: Board, max_depth: int, time_limit_s: float) -> Tuple[Move, int]:
        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = max(0.0, time_limit_s)

        best_move: Optional[Move] = None
        best_score = -INF
        prev_score = 0

        for depth in range(1, max_depth + 1):
            try:
                # aspiration window around previous
                window = 60 if depth >= 3 else 200
                alpha = prev_score - window
                beta = prev_score + window

                while True:
                    score = self.negamax(b, depth, alpha, beta, 0)
                    if score <= alpha:
                        alpha -= window
                        window *= 2
                        continue
                    if score >= beta:
                        beta += window
                        window *= 2
                        continue
                    break

                prev_score = score

                # best move from TT root
                root = self.tt.probe(b.key)
                if root is not None and root.move is not None:
                    best_move = root.move
                    best_score = score

                elapsed = max(1e-6, time.time() - self.start_time)
                nps = int(self.nodes / elapsed)

                score_str = f"score cp {best_score}"
                if abs(best_score) > MATE - 1000:
                    mate_in = (MATE - abs(best_score) + 1) // 2
                    score_str = f"score mate {mate_in if best_score > 0 else -mate_in}"

                pv = self._extract_pv(b, best_move, depth)
                print(f"info depth {depth} {score_str} nodes {self.nodes} nps {nps} time {int(elapsed*1000)} pv {pv}")
                sys.stdout.flush()

            except SearchTimeout:
                break

        if best_move is None:
            lm = b.legal_moves()
            best_move = lm[0] if lm else Move(0, 0)
            best_score = 0

        return best_move, best_score

    def _extract_pv(self, b: Board, first: Optional[Move], depth: int) -> str:
        if first is None:
            return ""
        pv = []
        seen = set()
        bb = Board()
        bb.set_fen(b.fen())
        mv = first
        for _ in range(depth):
            pv.append(mv.uci())
            undo = bb.push(mv)
            if bb.key in seen:
                bb.pop(undo)
                break
            seen.add(bb.key)
            e = self.tt.probe(bb.key)
            if e is None or e.move is None:
                bb.pop(undo)
                break
            mv = e.move
            bb.pop(undo)
        return " ".join(pv)


# -----------------------------
# UCI loop
# -----------------------------

def uci_loop() -> None:
    eng = Engine()
    b = Board()
    b.set_startpos()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()

        if cmd == "uci":
            print("id name GeoConflictModeler-UCI-Permissive")
            print("id author warcolor")
            print("option name Hash type spin default 128 min 1 max 1024")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            b.set_startpos()

        elif cmd.startswith("setoption"):
            parts = cmd.split()
            if "name" in parts and "value" in parts:
                ni = parts.index("name") + 1
                vi = parts.index("value") + 1
                name = " ".join(parts[ni:vi - 1]).strip().lower()
                val = " ".join(parts[vi:]).strip()
                if name == "hash":
                    try:
                        eng.set_hash_mb(int(val))
                    except Exception:
                        pass

        elif cmd.startswith("position"):
            parts = cmd.split()
            if len(parts) >= 2 and parts[1] == "startpos":
                b.set_startpos()
                if "moves" in parts:
                    mi = parts.index("moves")
                    for ms in parts[mi + 1:]:
                        mv = parse_uci_move(b, ms)
                        # ensure legality
                        lm = b.legal_moves()
                        ok = False
                        for m in lm:
                            if m.uci() == mv.uci():
                                b.push(m)
                                ok = True
                                break
                        if not ok:
                            break
            elif len(parts) >= 2 and parts[1] == "fen":
                fen = " ".join(parts[2:8])
                b.set_fen(fen)
                if "moves" in parts:
                    mi = parts.index("moves")
                    for ms in parts[mi + 1:]:
                        mv = parse_uci_move(b, ms)
                        lm = b.legal_moves()
                        ok = False
                        for m in lm:
                            if m.uci() == mv.uci():
                                b.push(m)
                                ok = True
                                break
                        if not ok:
                            break

        elif cmd.startswith("go"):
            parts = cmd.split()
            depth = 6
            movetime_ms: Optional[int] = None
            wtime = btime = winc = binc = None

            def get_int(tag: str) -> Optional[int]:
                if tag in parts:
                    try:
                        return int(parts[parts.index(tag) + 1])
                    except Exception:
                        return None
                return None

            if "depth" in parts:
                depth = get_int("depth") or 6
            movetime_ms = get_int("movetime")
            wtime = get_int("wtime")
            btime = get_int("btime")
            winc = get_int("winc") or 0
            binc = get_int("binc") or 0

            time_limit_s = 0.0
            if movetime_ms is not None:
                time_limit_s = max(0.0, (movetime_ms - 20) / 1000.0)
            elif wtime is not None and btime is not None:
                remaining = wtime if b.turn == 1 else btime
                inc = winc if b.turn == 1 else binc
                budget = remaining / 30.0 + inc * 0.8
                time_limit_s = max(0.0, (budget - 20) / 1000.0)
                time_limit_s = min(time_limit_s, max(0.05, remaining / 1000.0 * 0.2))

            best, _ = eng.search(b, max_depth=depth, time_limit_s=time_limit_s)
            print(f"bestmove {best.uci()}")
            sys.stdout.flush()

        elif cmd == "quit":
            break


def parse_uci_move(b: Board, s: str) -> Move:
    s = s.strip()
    frm = parse_uci_sq(s[0:2])
    to = parse_uci_sq(s[2:4])
    promo = 0
    if len(s) >= 5:
        ch = s[4].lower()
        promo = {'q': Q, 'r': R, 'b': B, 'n': N}.get(ch, 0)
    return Move(frm, to, promo)


if __name__ == "__main__":
    uci_loop()
