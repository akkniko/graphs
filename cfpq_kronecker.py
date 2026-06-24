"""
CFPQ — алгоритм на основе произведения Кронекера (тензорного произведения).
из диссертации Азимова: Глава 4, раздел 4.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, kron, lil_matrix
from scipy.sparse.csgraph import connected_components



@dataclass
class RSMBox:
    """
    states: список состояний/индексы
    start:  начальное состояние
    finals: множество конечных состояний
    transitions: (src_state, label, dst_state)
    """
    head: str
    num_states: int
    start: int
    finals: Set[int]
    transitions: List[Tuple[int, str, int]] = field(default_factory=list)


@dataclass
class RSM:
    boxes: Dict[str, RSMBox] = field(default_factory=dict)
    start_nonterminal: str = "S"

    @classmethod
    def from_grammar_text(cls, text: str, start: str = "S") -> "RSM":

        rsm = cls(start_nonterminal=start)
        rules_by_head: Dict[str, List[List[str]]] = {}

        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, _, body_str = line.partition("->")
            head = head.strip()
            body = body_str.strip().split()
            rules_by_head.setdefault(head, []).append(body)

        for head, rules in rules_by_head.items():
            # Состояния: 0 - глобальный старт ящика
            # Для каждого правила - своя цепочка состояний
            transitions: List[Tuple[int, str, int]] = []
            finals: Set[int] = set()
            state_counter = 1  # 0 — старт

            for body in rules:
                if body == ["eps"]:
                    finals.add(0)  # ε-правило: старт = финиш
                    continue
                prev = 0
                for sym in body:
                    cur = state_counter
                    state_counter += 1
                    transitions.append((prev, sym, cur))
                    prev = cur
                finals.add(prev)

            box = RSMBox(
                head=head,
                num_states=state_counter,
                start=0,
                finals=finals,
                transitions=transitions,
            )
            rsm.boxes[head] = box

        return rsm


# Алгоритм на основе произведения Кронекера
def cfpq_kronecker(
    graph_edges: List[Tuple[int, str, int]],
    num_nodes: int,
    grammar_text: str,
    start_nonterminal: str = "S",
) -> List[Tuple[int, int]]:
    """
    Контекстно-свободный поиск путей через произведение Кронекера.

    graph_edges         : рёбра графа (src, label, dst)
    num_nodes           : число вершин в графе
    grammar_text        : грамматика в виде текста (произвольные правила, не CNF)
    start_nonterminal   : стартовый нетерминал

    return:Список пар (i, j) = вершины, достижимые по стартовому нетерминалу
    """
    t0 = time.perf_counter()

    rsm = RSM.from_grammar_text(grammar_text, start_nonterminal)

    #Сбор всех символов (терминалы и нетерминалы)
    all_labels: Set[str] = set()
    for src, lbl, dst in graph_edges:
        all_labels.add(lbl)
    for box in rsm.boxes.values():
        for _, lbl, _ in box.transitions:
            all_labels.add(lbl)

    n = num_nodes  # число вершин графа

    graph_matrices: Dict[str, lil_matrix] = {
        lbl: lil_matrix((n, n), dtype=np.bool_) for lbl in all_labels
    }
    for src, lbl, dst in graph_edges:
        graph_matrices[lbl][src, dst] = True
    graph_csr: Dict[str, csr_matrix] = {
        lbl: m.tocsr() for lbl, m in graph_matrices.items()
    }

    state_offset: Dict[str, int] = {}
    total_rsm_states = 0
    for nt, box in rsm.boxes.items():
        state_offset[nt] = total_rsm_states
        total_rsm_states += box.num_states

    Q = total_rsm_states  # общее число состояний RSM

    rsm_matrices: Dict[str, lil_matrix] = {
        lbl: lil_matrix((Q, Q), dtype=np.bool_) for lbl in all_labels
    }
    for nt, box in rsm.boxes.items():
        off = state_offset[nt]
        for (s, lbl, d) in box.transitions:
            rsm_matrices[lbl][off + s, off + d] = True
    rsm_csr: Dict[str, csr_matrix] = {
        lbl: m.tocsr() for lbl, m in rsm_matrices.items()
    }

    # Итеративно строится кронекерово произведение и транзитивное замыкание
    def build_combined_matrix() -> csr_matrix:
        combined = csr_matrix((Q * n, Q * n), dtype=np.bool_)
        for lbl in all_labels:
            R = rsm_csr.get(lbl)
            M = graph_csr.get(lbl)
            if R is None or M is None:
                continue
            if R.nnz == 0 or M.nnz == 0:
                continue
            combined = combined + kron(R, M, format="csr").astype(np.bool_)
        return combined

    def transitive_closure(mat: csr_matrix) -> csr_matrix:
        result = mat.astype(np.bool_)
        prev_nnz = -1
        while result.nnz != prev_nnz:
            prev_nnz = result.nnz
            step = result @ result
            result = (result + step).astype(np.bool_)
            result.eliminate_zeros()
        return result

    changed = True
    iterations = 0
    while changed:
        combined = build_combined_matrix()
        tc = transitive_closure(combined)

        # Извлекаем новые рёбра для нетерминалов
        changed = False
        for nt, box in rsm.boxes.items():
            off = state_offset[nt]
            q_start = off + box.start
            new_lil = lil_matrix((n, n), dtype=np.bool_)
            for q_fin in box.finals:
                q_f = off + q_fin
                # Поиск пар (q_start * n + v, q_f * n + u) в tc
                # => ребро v->u помечено нетерминалом nt
                for v in range(n):
                    row_idx = q_start * n + v
                    for u in range(n):
                        col_idx = q_f * n + u
                        if tc[row_idx, col_idx]:
                            if not graph_csr.get(nt, csr_matrix((n, n)))[v, u]:
                                new_lil[v, u] = True
                                changed = True

            if changed:
                old = graph_csr.get(nt, csr_matrix((n, n), dtype=np.bool_))
                graph_csr[nt] = (old + new_lil.tocsr()).astype(np.bool_)
                rsm_csr[nt] = rsm_csr.get(nt, csr_matrix((Q, Q), dtype=np.bool_))

        iterations += 1

    # рез-т: пары вершин для стартового нетерминала
    start_box = rsm.boxes.get(start_nonterminal)
    result_pairs: List[Tuple[int, int]] = []

    if start_box is not None:
        off = state_offset[start_nonterminal]
        q_start = off + start_box.start
        final_mat = build_combined_matrix()
        tc_final = transitive_closure(final_mat)

        for q_fin in start_box.finals:
            q_f = off + q_fin
            for v in range(n):
                for u in range(n):
                    if tc_final[q_start * n + v, q_f * n + u]:
                        result_pairs.append((v, u))

        if start_box.start in start_box.finals:
            for v in range(n):
                if (v, v) not in result_pairs:
                    result_pairs.append((v, v))

    elapsed = time.perf_counter() - t0
    print(
        f"[CFPQ-Kronecker] n={n}, итераций={iterations}, "
        f"пар={len(result_pairs)}, время={elapsed:.4f}s"
    )
    return list(set(result_pairs))