"""
!!Исследование 2!!

Алгоритм пофазный:
    Фаза 1. Запустить CFPQ только для правил 'a'
            Получить матрицу T_a: T_a[i,j]=1 => i достижим из i по a-цепочке
    Фаза 2. Из T_a извлечь граничные узлы - вершины j, достижимые
            по a-цепочке из хотя бы одной вершины. Это кандидаты на
            переход a->b
    Фаза 3. Запустить CFPQ для b-правил только от граничных узлов.
            Получить T_b
    Фаза 4. Повторить аналогично для второго блока a и b
    Фаза 5. Финальная проверка: существует ли вершина i такая, что
            i ->a-> j ->b-> k ->a-> l ->b-> i
"""

'''
пример команды для запуска программы
   python research/cycle_abab.py graphs/abab_cycle.csv    a b
    python research/cycle_abab.py graphs/abab_no_cycle.csv a b
    python research/cycle_abab.py graphs/paper_double_cycle.csv a b
'''
import sys
import os
import time
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

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



def transitive_closure_single_label(
    edges: List[Tuple[int, str, int]],
    num_nodes: int,
    label: str,
    restrict_sources: Optional[Set[int]] = None,
) -> csr_matrix:
    
    n = num_nodes
    mat = lil_matrix((n, n), dtype=np.bool_)

    for src, lbl, dst in edges:
        if lbl != label:
            continue
        if restrict_sources is not None and src not in restrict_sources:
            continue
        mat[src, dst] = True

    # Транзитивное замыкание: T = T | T@T до фиксации
    result = mat.tocsr().astype(np.bool_)
    prev_nnz = -1
    iters = 0
    while result.nnz != prev_nnz:
        prev_nnz = result.nnz
        step = result @ result
        result = (result + step).astype(np.bool_)
        result.eliminate_zeros()
        iters += 1

    return result, iters

def find_abab_cycles(
    graph_path: str,
    label_a: str = "a",
    label_b: str = "b",
    node_names: Optional[Dict[int, str]] = None,
) -> List[Tuple[int, int, int, int]]:

    edges, num_nodes = load_graph(graph_path)
    n = num_nodes
    print(f"  Граф: {graph_path}  ({n} вершин, {len(edges)} рёбер)")
    print(f"  Метки: a='{label_a}',  b='{label_b}'")
    print()

    stats = {}  # для замеров

    #Фаза 1: T_a - все a-достижимости
    t0 = time.perf_counter()
    T_a, iters_a = transitive_closure_single_label(edges, n, label_a)
    stats["phase1_time"] = time.perf_counter() - t0
    stats["phase1_iters"] = iters_a
    stats["phase1_nnz"] = T_a.nnz
    print(f"  Фаза 1 (T_a): nnz={T_a.nnz}, итераций={iters_a}, время={stats['phase1_time']:.4f}с")

    # Граничные узлы после первого блока 'a':
    # это все j такие, что существует i: T_a[i,j] = 1
    a_targets: Set[int] = set(T_a.tocoo().col.tolist())
    print(f"  Граничные вершины a→b: {sorted(a_targets)}")

    if not a_targets:
        print("\n  Нет a-достижимостей → циклов быть не может.")
        return []

    #Фаза 2: T_b1 - b-достижимости только из a_targets 
    t0 = time.perf_counter()
    T_b1, iters_b1 = transitive_closure_single_label(edges, n, label_b, restrict_sources=a_targets)
    stats["phase2_time"] = time.perf_counter() - t0
    stats["phase2_iters"] = iters_b1
    stats["phase2_nnz"] = T_b1.nnz
    print(f"  Фаза 2 (T_b1): nnz={T_b1.nnz}, итераций={iters_b1}, время={stats['phase2_time']:.4f}с")

    b1_targets: Set[int] = set(T_b1.tocoo().col.tolist())
    print(f"  Граничные вершины b→a: {sorted(b1_targets)}")

    if not b1_targets:
        print("\n  Нет b-достижимостей → циклов быть не может.")
        return []

    #Фаза 3: T_a2 - второй блок a от b1_targets 
    t0 = time.perf_counter()
    T_a2, iters_a2 = transitive_closure_single_label(edges, n, label_a, restrict_sources=b1_targets)
    stats["phase3_time"] = time.perf_counter() - t0
    stats["phase3_iters"] = iters_a2
    stats["phase3_nnz"] = T_a2.nnz
    print(f"  Фаза 3 (T_a2): nnz={T_a2.nnz}, итераций={iters_a2}, время={stats['phase3_time']:.4f}с")

    a2_targets: Set[int] = set(T_a2.tocoo().col.tolist())
    print(f"  Граничные вершины a→b (второй раз): {sorted(a2_targets)}")

    if not a2_targets:
        print("\n  Нет второго a-блока → циклов быть не может.")
        return []

    # Фаза 4: T_b2 — второй блок b от a2_targets
    t0 = time.perf_counter()
    T_b2, iters_b2 = transitive_closure_single_label(edges, n, label_b, restrict_sources=a2_targets)
    stats["phase4_time"] = time.perf_counter() - t0
    stats["phase4_iters"] = iters_b2
    stats["phase4_nnz"] = T_b2.nnz
    print(f"  Фаза 4 (T_b2): nnz={T_b2.nnz}, итераций={iters_b2}, время={stats['phase4_time']:.4f}с")

    print()
    total_time = sum(v for k, v in stats.items() if k.endswith("_time"))
    print(f"  Суммарное время фаз: {total_time:.4f}с")
    
    #
    #Фаза 5: финальная проверка циклов O(n) 
    t0 = time.perf_counter()
    combined = T_a.astype(np.bool_) @ T_b1.astype(np.bool_) @ T_a2.astype(np.bool_) @ T_b2.astype(np.bool_)
    combined = combined.astype(np.bool_)
    combined.eliminate_zeros()
    check_time = time.perf_counter() - t0
    print(f"  Финальная проверка (T_a @ T_b1 @ T_a2 @ T_b2): время={check_time:.4f}с, nnz={combined.nnz}")

    # Диагональ комбинированной матрицы
    diag = combined.diagonal()
    cycle_nodes = [i for i in range(n) if diag[i]]

    print()
    if cycle_nodes:
        print(f"  x ОБНАРУЖЕНЫ ЦИКЛЫ a^m b^k a^i b^j  ({len(cycle_nodes)} вершин в цикле):")
        for v in cycle_nodes:
            name = node_names.get(v, str(v)) if node_names else str(v)
            print(f"      вершина {v} ({name}) участвует в цикле")
    else:
        print("  + ЦИКЛОВ a^m b^k a^i b^j НЕТ.")

    # Восстановление конкретных путей для найденных цикличных вершин
    cycles_found = []
    if cycle_nodes:
        print()
        print("========Восстановление структуры цикла ======")
        print("  (ищем конкретные вершины i0->i1->i2->i3->i0)")

        cx_a   = T_a.tocoo()
        cx_b1  = T_b1.tocoo()
        cx_a2  = T_a2.tocoo()
        cx_b2  = T_b2.tocoo()

        pairs_a  = set(zip(cx_a.row,  cx_a.col))
        pairs_b1 = set(zip(cx_b1.row, cx_b1.col))
        pairs_a2 = set(zip(cx_a2.row, cx_a2.col))
        pairs_b2 = set(zip(cx_b2.row, cx_b2.col))

        for i0 in cycle_nodes:
            # Найти один конкретный цикл через i0
            found = False
            for i1 in range(n):
                if (i0, i1) not in pairs_a:
                    continue
                for i2 in range(n):
                    if (i1, i2) not in pairs_b1:
                        continue
                    for i3 in range(n):
                        if (i2, i3) not in pairs_a2:
                            continue
                        if (i3, i0) in pairs_b2:
                            n0 = node_names.get(i0, str(i0)) if node_names else str(i0)
                            n1 = node_names.get(i1, str(i1)) if node_names else str(i1)
                            n2 = node_names.get(i2, str(i2)) if node_names else str(i2)
                            n3 = node_names.get(i3, str(i3)) if node_names else str(i3)
                            print(f"      {n0} -a^m-> {n1} -b^k-> {n2} -a^i-> {n3} -b^j-> {n0}")
                            cycles_found.append((i0, i1, i2, i3))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

    return cycles_found


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        script_name = Path(sys.argv[0]).name
        graphs_dir = PROJECT_ROOT / "graphs"
        
        print(f"Использование: python research/{script_name} <граф.csv> [метка_a] [метка_b]")
        print()
        print("Примеры:")
        print(f"  python research\{script_name} graphs\paper_double_cycle.csv a b")
        print(f"  python research\{script_name} graphs\object_model.csv a b")
        print(f"  python research\{script_name} graphs\same_generation.csv a b")
        sys.exit(0)

    graph_path = sys.argv[1]
    label_a    = sys.argv[2] if len(sys.argv) > 2 else "a"
    label_b    = sys.argv[3] if len(sys.argv) > 3 else "b"

    print("=" * 60)
    print("  Исследование 2: поиск циклов a^m b^k a^i b^j")
    print("=" * 60)
    print()

    find_abab_cycles(graph_path, label_a, label_b)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
