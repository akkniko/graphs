"""
CFPQ — алгоритм на основе матричного умножения
Алгоритм:
  1. Строится матрица T размера |V|×|V|, элементы — множества нетерминалов.
  2. Инициализируется T по рёбрам графа и правилам вида A → x.
  3. Итеративно: T = T ∪ (T × T), пока T не перестанет меняться.
  4. Результат: T^cf[i][j] содержит нетерминал A iff (i,j) ∈ R_A.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

# Типы данных
@dataclass
class Grammar:
    """
    Правила трёх видов:
      A → B C    (binary_rules)
      A → x      (terminal_rules)
      A → ε      (epsilon_nonterminals)
    """
    nonterminals: Set[str] = field(default_factory=set)
    terminals: Set[str] = field(default_factory=set)
    binary_rules: List[Tuple[str, str, str]] = field(default_factory=list)   # (A, B, C)
    terminal_rules: List[Tuple[str, str]] = field(default_factory=list)      # (A, x)
    epsilon_nonterminals: Set[str] = field(default_factory=set)              # A → ε
    start: str = "S"

    @classmethod
    def from_text(cls, text: str) -> "Grammar":
        """
        Простой парсер правил.
        Формат каждой строки:
            A -> B C      (бинарное)
            A -> x        (терминальное, если x в нижнем регистре)
            A -> eps      (эпсилон)
        Символы в верхнем регистре считаются нетерминалами.
        """
        g = cls()
        for raw in text.strip().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, _, body_str = line.partition("->")
            head = head.strip()
            body = body_str.strip().split()
            g.nonterminals.add(head)
            if not body or body == ["eps"]:
                g.epsilon_nonterminals.add(head)
            elif len(body) == 2:
                B, C = body
                g.nonterminals.update([B, C])
                g.binary_rules.append((head, B, C))
            elif len(body) == 1:
                x = body[0]
                g.terminals.add(x)
                g.terminal_rules.append((head, x))
            else:
                raise ValueError(f"Правило не в WCNF: {line!r}")
        return g


@dataclass
class LabeledGraph:
    num_nodes: int
    edges: List[Tuple[int, str, int]] = field(default_factory=list)

    @classmethod
    def from_edges(cls, edges: List[Tuple[int, str, int]]) -> "LabeledGraph":
        if not edges:
            return cls(num_nodes=0, edges=[])
        n = max(max(u, v) for u, _, v in edges) + 1
        return cls(num_nodes=n, edges=edges)


class SetMatrix:
    def __init__(self, n: int):
        self.n = n
        self._data: List[List[Set[str]]] = [[set() for _ in range(n)] for _ in range(n)]

    def __getitem__(self, idx):
        i, j = idx
        return self._data[i][j]

    def add(self, i: int, j: int, nt: str) -> bool:
        """Добавяляется нетерминал, возврат тру, если матрица изменилась"""
        if nt not in self._data[i][j]:
            self._data[i][j].add(nt)
            return True
        return False

    def multiply_update(self,binary_rules: List[Tuple[str, str, str]],) -> bool:
        n = self.n
        changed = False
        bc_to_a: Dict[Tuple[str, str], List[str]] = {}
        for A, B, C in binary_rules:
            bc_to_a.setdefault((B, C), []).append(A)

        new_entries: List[Tuple[int, int, str]] = []
        for i in range(n):
            for k in range(n):
                if not self._data[i][k]:
                    continue
                for j in range(n):
                    if not self._data[k][j]:
                        continue
                    for B in self._data[i][k]:
                        for C in self._data[k][j]:
                            if (B, C) in bc_to_a:
                                for A in bc_to_a[(B, C)]:
                                    if A not in self._data[i][j]:
                                        new_entries.append((i, j, A))

        for i, j, A in new_entries:
            if self.add(i, j, A):
                changed = True
        return changed

    def get_pairs(self, nonterminal: str) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for i in range(self.n)
            for j in range(self.n)
            if nonterminal in self._data[i][j]
        ]


class SparseSetMatrix:
    """
    Матрица n×n, где для каждого нетерминала хранится отдельная
    булева разреженная матрица (CSR). Это T_A для каждого A из N
    т.е. T_A |= T_B @ T_C     (булево умножение)
    """

    def __init__(self, n: int, nonterminals: Set[str]):
        self.n = n
        self.nonterminals = nonterminals
        self._matrices: Dict[str, lil_matrix] = {nt: lil_matrix((n, n), dtype=np.bool_) for nt in nonterminals}

    def add(self, nt: str, i: int, j: int) -> bool:
        if self._matrices[nt][i, j]:
            return False
        self._matrices[nt][i, j] = True
        return True

    def to_csr(self) -> Dict[str, csr_matrix]:
        return {nt: m.tocsr() for nt, m in self._matrices.items()}

    def update_from_csr(self, csr_dict: Dict[str, csr_matrix]) -> bool:
        changed = False
        for nt, new_mat in csr_dict.items():
            old = self._matrices[nt]
            cx = new_mat.tocoo()
            for i, j in zip(cx.row, cx.col):
                if not old[i, j]:
                    old[i, j] = True
                    changed = True
        return changed

    def get_pairs(self, nonterminal: str) -> List[Tuple[int, int]]:
        m = self._matrices[nonterminal].tocsr()
        cx = m.tocoo()
        return list(zip(cx.row.tolist(), cx.col.tolist()))



def cfpq_matrix(
    graph: LabeledGraph,
    grammar: Grammar,
    start_nonterminal: Optional[str] = None,
    use_sparse: bool = True,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    поиск путей методом матричного умножения

    graph               : граф с метками рёбер
    grammar             : КС-грамматика в WCNF
    start_nonterminal   : нетерминал-старт 
    use_sparse          : использовать разреженные матрицы 

    erturn:
    Словарь {нетерминал: [(i,j), ...]} — пары вершин для каждого нетерминала.
    """
    if start_nonterminal is None:
        start_nonterminal = grammar.start

    n = graph.num_nodes
    if n == 0:
        return {nt: [] for nt in grammar.nonterminals}

    t0 = time.perf_counter()

    if use_sparse:
        result = _cfpq_sparse(graph, grammar)
    else:
        result = _cfpq_dense(graph, grammar)

    elapsed = time.perf_counter() - t0
    print(f"[CFPQ-Matrix] n={n}, итераций выполнено, время={elapsed:.4f}s")
    return result


def _cfpq_dense(graph: LabeledGraph,grammar: Grammar,) -> Dict[str, List[Tuple[int, int]]]:
    """Реализация через SetMatrix (прозрачная, для небольших графов)."""
    n = graph.num_nodes
    T = SetMatrix(n)

    terminal_map: Dict[str, List[str]] = {}
    for A, x in grammar.terminal_rules:
        terminal_map.setdefault(x, []).append(A)

    for src, label, dst in graph.edges:
        for A in terminal_map.get(label, []):
            T.add(src, dst, A)

    for A in grammar.epsilon_nonterminals:
        for v in range(n):
            T.add(v, v, A)

    iterations = 0
    while True:
        changed = T.multiply_update(grammar.binary_rules)
        iterations += 1
        if not changed:
            break

    print(f"  [dense] итераций: {iterations}")
    return {nt: T.get_pairs(nt) for nt in grammar.nonterminals}


def _cfpq_sparse(graph: LabeledGraph,grammar: Grammar,) -> Dict[str, List[Tuple[int, int]]]:
   
    n = graph.num_nodes
    SM = SparseSetMatrix(n, grammar.nonterminals)

    terminal_map: Dict[str, List[str]] = {}
    for A, x in grammar.terminal_rules:
        terminal_map.setdefault(x, []).append(A)

    for src, label, dst in graph.edges:
        for A in terminal_map.get(label, []):
            SM.add(A, src, dst)

    for A in grammar.epsilon_nonterminals:
        for v in range(n):
            SM.add(A, v, v)

    iterations = 0
    while True:
        csr = SM.to_csr()
        new_entries: Dict[str, csr_matrix] = {}

        for A, B, C in grammar.binary_rules:
            product = csr[B].astype(np.bool_) @ csr[C].astype(np.bool_)
            product = product.astype(np.bool_)
            diff = product - csr[A].astype(np.bool_)
            diff.eliminate_zeros()
            if diff.nnz > 0:
                new_entries[A] = new_entries.get(A, csr_matrix((n, n), dtype=np.bool_)) + diff

        if not new_entries:
            break

        changed = SM.update_from_csr(new_entries)
        iterations += 1
        if not changed:
            break

    print(f"  [sparse] итераций: {iterations}")
    return {nt: SM.get_pairs(nt) for nt in grammar.nonterminals}