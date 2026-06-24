"""
Исследование 1 - проверка цикла типа a^n в графе

Идея: если класс A через цепочку рёбер типа 'a' (например subclassOf)
достигает сам себя, то в результирующей матрице T_S[A, A] = 1,
то есть нетерминал S присутствует на диагонали матрицы.
"""


'''
пример команды запуска программы
python research/check_diagonal.py graphs/inheritance_cycle.csv grammars/subclassof_cnf.txt S
python research/check_diagonal.py graphs/inheritance_ok.csv    grammars/subclassof_cnf.txt S
python research/check_diagonal.py graphs/paper_double_cycle.csv grammars/anbn_cnf.txt S
'''

import sys
import os
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfpq_matrix import Grammar, LabeledGraph, cfpq_matrix
from grammar_utils import parse_and_normalize


def load_graph(path: str) -> Tuple[List[Tuple[int, str, int]], int]:
    edges, max_node = [], -1
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            src, label, dst = int(parts[0]), parts[1].strip(), int(parts[2])
            edges.append((src, label, dst))
            max_node = max(max_node, src, dst)
    return edges, max_node + 1


def load_grammar(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def check_diagonal(
    graph_path: str,
    grammar_path: str,
    start_nt: str = "S",
    node_names: Optional[Dict[int, str]] = None,
) -> List[int]:
    """
    Запуск CFPQ, извлечение матрицы T_S и проверка диагонали

    Возвращает список индексов вершин, у которых T_S[i,i] = 1,
    т.е.  вершин, участвующих в цикле по языку грамматики
    """
    edges, num_nodes = load_graph(graph_path)
    grammar_text = load_grammar(grammar_path)
    grammar = parse_and_normalize(grammar_text, start=start_nt)
    graph = LabeledGraph(num_nodes=num_nodes, edges=edges)

    print(f"  Граф:       {graph_path}  ({num_nodes} вершин, {len(edges)} рёбер)")
    print(f"  Грамматика: {grammar_path}  (старт: {start_nt})")
    print()

    t0 = time.perf_counter()
    results = cfpq_matrix(graph, grammar, start_nonterminal=start_nt, use_sparse=True)
    elapsed = time.perf_counter() - t0

    pairs = results.get(start_nt, [])

    diagonal_nodes = sorted(i for (i, j) in pairs if i == j)

    print(f"  Время CFPQ:          {elapsed:.4f} с")
    print(f"  Всего пар в R_{start_nt}: {len(pairs)}")
    print()

    if diagonal_nodes:
        print(f"  x ОБНАРУЖЕНЫ ЦИКЛЫ ({len(diagonal_nodes)} вершин):")
        for v in diagonal_nodes:
            name = node_names.get(v, str(v)) if node_names else str(v)
            print(f"      вершина {v} ({name}) достигает сама себя")
        print()
        print("  Вывод: граф содержит запрещённый цикл по правилам грамматики.")
    else:
        print("  + ЦИКЛОВ НЕТ - диагональ T_S нулевая.")
        print("  Вывод: ни одна вершина не достигает себя по правилам грамматики.")

    print()
    print("  Все найденные пары R_S:")
    for i, j in sorted(pairs):
        ni = node_names.get(i, str(i)) if node_names else str(i)
        nj = node_names.get(j, str(j)) if node_names else str(j)
        marker = "  ← ЦИКЛ" if i == j else ""
        print(f"      ({ni}, {nj}){marker}")

    return diagonal_nodes


#какие конкретно цепочки образуют цикл
def trace_cycles(
    graph_path: str,
    grammar_path: str,
    start_nt: str = "S",
    node_names: Optional[Dict[int, str]] = None,
    max_depth: int = 10,
) -> None:
    edges, num_nodes = load_graph(graph_path)
    cyclic_nodes = check_diagonal(graph_path, grammar_path, start_nt, node_names)

    if not cyclic_nodes:
        return

    print()
    print("===========Трассировка путей, образующих циклы=======================")

    #словарь смежности
    adj: Dict[int, List[Tuple[str, int]]] = {i: [] for i in range(num_nodes)}
    for src, lbl, dst in edges:
        adj[src].append((lbl, dst))

    for start in cyclic_nodes:
        name = node_names.get(start, str(start)) if node_names else str(start)
        print(f"\n  Цикл из вершины {start} ({name}):")

        # BFS поиск пути обратно в start
        from collections import deque
        queue = deque([(start, [])])
        visited = set()
        found = False

        while queue and not found:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for lbl, nxt in adj[node]:
                new_path = path + [(lbl, nxt)]
                if nxt == start and new_path:
                    # Нашли цикл
                    route = f"{start}"
                    for l, n in new_path:
                        route += f" --{l}--> {n}"
                    print(f"      {route}")
                    found = True
                    break
                key = (nxt, tuple(p[0] for p in new_path))
                if key not in visited:
                    visited.add(key)
                    queue.append((nxt, new_path))

        if not found:
            print(f"      (путь длиннее {max_depth} шагов или не найден простым BFS)")



def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 3:
        print("Использование: python research/check_diagonal.py <граф.csv> <грамматика.txt> [нетерминал]")
        print()
        print("Примеры:")
        print("  python research/check_diagonal.py graphs/inheritance_cycle.csv grammars/subclassof_cnf.txt S")
        print("  python research/check_diagonal.py graphs/inheritance_ok.csv    grammars/subclassof_cnf.txt S")
        print("  python research/check_diagonal.py graphs/paper_double_cycle.csv grammars/anbn_cnf.txt S")
        sys.exit(0)

    graph_path   = sys.argv[1]
    grammar_path = sys.argv[2]
    start_nt     = sys.argv[3] if len(sys.argv) > 3 else "S"

    print("=" * 55)
    print("  Исследование 1: проверка цикла a^n (диагональ T_S)")
    print("=" * 55)
    print()

    trace_cycles(graph_path, grammar_path, start_nt)

    print()
    print("=" * 55)


if __name__ == "__main__":
    main()
