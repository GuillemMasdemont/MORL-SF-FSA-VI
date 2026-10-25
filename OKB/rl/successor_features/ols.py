import random
from typing import List, Optional, Union

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

import wandb as wb
from rl.utils.eval import hypervolume, policy_evaluation_mo
from rl.utils.utils import extrema_weights, random_weights, unique_tol, remove_array_from_list, equally_spaced_weights, incremental_weights

np.set_printoptions(precision=4)


class OLS:
    def __init__(
        self,
        m: int,
        epsilon: Optional[float] = None,
        negative_weights: bool = False,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        reverse_extremum: bool = False,
        sample_k: Optional[int] = None,
    ):
        self.m = m
        self.epsilon = epsilon
        self.W = []
        self.ccs = []
        self.ccs_weights = []
        self.queue = []
        self.iteration = 0
        self.max_value = max_value
        self.min_value = min_value
        self.negative_weights = negative_weights
        self.sample_k = sample_k
        self.worst_case_weight_repeated = False
        self.not_improved_weights = []
        extremum_weights = reversed(self.extrema_weights()) if reverse_extremum else self.extrema_weights()
        for w in extremum_weights:
            self.queue.append((float("inf"), w))

    def next_w(self, gpi_agent=None, env=None, no_priority=False, rep_eval=1) -> np.ndarray:
        if len(self.ccs) > 0:
            if self.sample_k is not None:
                W_corner = self.sample_corner_weights(self.sample_k)
            else:
                W_corner = self.corner_weights_new()

            if gpi_agent is not None:
                gpi_expanded_set = [policy_evaluation_mo(gpi_agent, env, wc, rep=rep_eval) for wc in W_corner]
            else:
                gpi_expanded_set = None

            print("W_corner size:", len(W_corner))
            self.queue = []
            for wc in W_corner:
                if no_priority:
                    priority = 0.0
                else:
                    priority = self.get_priority(wc, gpi_expanded_set)
                if self.epsilon is None or priority >= self.epsilon:
                    if not any([np.allclose(wc, wold) for wold in self.not_improved_weights]):
                        self.queue.append((priority, wc.astype(np.float32)))

            if len(self.queue) > 0:
                self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority
                if self.queue[0][0] == 0.0:  # shuffle weights if all have priority 0
                    random.shuffle(self.queue)
                elif np.isinf(self.queue[0][0]):  # if there are infinity priorities, let the extreme weights be first
                    self.queue.sort(key=lambda t: t[1].max(), reverse=True)
                print("queue:", self.queue)
                print(f"gap ols: {self.queue[0][0]:.6f}, ols w: {self.queue[0][1]}")

        print("ccs:", self.ccs)
        print("ccs size:", len(self.ccs))

        if len(self.queue) == 0:
            return None
        return self.queue.pop(0)[1]

    def get_ccs_weights(self) -> List[np.ndarray]:
        return self.ccs_weights.copy()

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        weights = [w for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def add_solution(self, value, w, add_not_improved=True) -> List[int]:
        print("Adding value", value, "for weight", w)
        self.iteration += 1
        self.W.append(w)

        if self.is_dominated(value):
            print("Dominated value", value)
            if add_not_improved:
                self.not_improved_weights.append(w)
            return [len(self.ccs)]

        if add_not_improved and any([np.allclose(value, v) for v in self.ccs]):
            self.not_improved_weights.append(w)

        removed_indx = self.remove_obsolete_values(value)

        self.ccs.append(value)
        self.ccs_weights.append(w)
        self.W.append(w)
        self.W = unique_tol(self.W, tol=1e-5)
        print("CCS:", self.ccs, "CCS size:", len(self.ccs))
        print("Weight Support Set:", self.ccs_weights)

        return removed_indx

    def get_priority(self, w, gpi_expanded_set=None) -> float:
        def best_vector(values, w):
            max_v = values[0]
            for i in range(1, len(values)):
                if values[i] @ w > max_v @ w:
                    max_v = values[i]
            return max_v

        max_value_ccs = self.max_scalarized_value(w)
        if gpi_expanded_set is not None:
            max_value_gpi = best_vector(gpi_expanded_set, w)
            max_value_gpi = np.dot(max_value_gpi, w)
            priority = max_value_gpi - max_value_ccs
        else:
            max_optimistic_value = self.max_value_lp(w)
            priority = max_optimistic_value - max_value_ccs
        return priority

    def max_scalarized_value(self, w: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def get_set_max_policy_index(self, w: np.ndarray) -> int:
        if not self.ccs:
            return None
        return np.argmax([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []
        W_del = []
        inds_remove = []
        for i, (priority, cw) in enumerate(self.queue):
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):
                W_del.append(cw)
                inds_remove.append(i)
        for i in reversed(inds_remove):
            self.queue.pop(i)
        return W_del

    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        removed_indx = []
        for i in reversed(range(len(self.ccs))):
            i_ws = [w for w in self.W if np.dot(self.ccs[i], w) == self.max_scalarized_value(w) and np.dot(value, w) < np.dot(self.ccs[i], w)]
            if len(i_ws) == 0:
                print("removed value", self.ccs[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                removed_w = self.ccs_weights.pop(i)
                self.W = remove_array_from_list(self.W, removed_w)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        v = cp.Variable(self.m)
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Maximize(w @ v)
        constraints = [W @ v <= V]
        if self.max_value is not None:
            constraints.append(v <= self.max_value)
        if self.min_value is not None:
            constraints.append(v >= self.min_value)
        prob = cp.Problem(objective, constraints)
        try:
            value = prob.solve(verbose=False)
        except:
            try:
                value = prob.solve(solver=cp.SCIPY, verbose=False)
            except:
                try:
                    value = self.upper_bound_policy_caches(w_new)
                except:
                    print("Failed to solve max value lp")
                    value = -1.0
        return value

    def upper_bound_policy_caches(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        alpha = cp.Variable(len(self.W))
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Minimize(alpha @ V)
        constraints = [alpha @ W == w, alpha >= 0]
        prob = cp.Problem(objective, constraints)
        upper_bound = prob.solve()
        if prob.status == cp.OPTIMAL:
            return upper_bound
        else:
            return float("inf")

    def worst_case_weight(self) -> np.ndarray:
        if len(self.W) == 0:
            return random_weights(dim=self.m)
        w = None
        min = float("inf")
        w_var = cp.Variable(self.m)
        params = []
        for v in self.ccs:
            p = cp.Parameter(self.m)
            p.value = v
            params.append(v)
        for i in range(len(self.ccs)):
            objective = cp.Minimize(w_var @ params[i])
            constraints = [0 <= w_var, cp.sum(w_var) == 1]
            # constraints = [cp.norm(w_var) - 1 <= 0, 0 <= w_var]
            for j in range(len(self.ccs)):
                if i != j:
                    constraints.append(w_var @ (params[j] - params[i]) <= 0)
            prob = cp.Problem(objective, constraints)
            value = prob.solve()
            if value < min and prob.status == cp.OPTIMAL:
                min = value
                w = w_var.value.copy()

        if np.allclose(w, self.W[-1]):
            self.worst_case_weight_repeated = True

        return w

    def sample_corner_weights(self, k):
        if len(self.ccs) == 0:
            return extrema_weights(dim=self.m)
        
        ccs = self.ccs.copy()
        ccs = unique_tol(ccs, tol=1e-4)
        ccs = np.vstack(ccs)

        A, b = self.get_linear_system_corner_weights(ccs)

        random_vectors = np.random.randn(20000, A.shape[1])
        directions = random_vectors / np.linalg.norm(random_vectors, axis=1, keepdims=True)
        incremental = np.array(incremental_weights(dim=A.shape[1])[:100])
        directions = np.concatenate([incremental, directions], axis=0)

        # Array to store extreme points
        extreme_points = []
        bounds = [(0, 1) for _ in range(A.shape[1] - 1)] + [(None, None)]
        for direction in directions:
            # Maximize in the chosen direction
            res = linprog(-direction, A_ub=A, b_ub=b, method='highs', bounds=bounds)
            if res.success:
                extreme_points.append(res.x[:-1])

            # Minimize in the chosen direction
            res = linprog(direction, A_ub=A, b_ub=b, method='highs', bounds=bounds)
            if res.success:
                extreme_points.append(res.x[:-1])
            
            extreme_points = list(np.unique(extreme_points, axis=0))

            if len(extreme_points) >= k:
                break

        return unique_tol(extreme_points + extrema_weights(dim=self.m), tol=1e-4)

    def get_k_corner_weights(self, k):
        if len(self.ccs) == 0:
            return extrema_weights(dim=self.m)
        
        ccs = self.ccs.copy()
        ccs = unique_tol(ccs, tol=1e-4)
        ccs = np.vstack(ccs)
        max_v = np.max(np.abs(ccs))
        if max_v > 1.0:
            ccs = ccs / max_v
        ccs = np.round(ccs, decimals=4)

        W_corner = []
        A, b = self.get_linear_system_corner_weights(ccs)
        for _ in range(10000):
            inds = random.sample(range(A.shape[0]), self.m + 1)
            A_ = A[inds,:]
            b_ = b[list(inds)]

            if np.linalg.det(A_) == 0:
                continue

            x = np.linalg.solve(A_, b_)

            if np.all(A @ x <= b + 1e-6):
                x = x[:-1]
                x = np.clip(x, 0.0, 1.0)
                x = x / x.sum()
                W_corner.append(x)

            W_corner = unique_tol(W_corner, tol=1e-4)
            if len(W_corner) == k:
                break
            
        return unique_tol(W_corner + extrema_weights(dim=self.m), tol=1e-4)

    def get_linear_system_corner_weights(self, ccs):
        # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
        # https://or.stackexchange.com/questions/4540/how-to-find-all-vertices-of-a-polyhedron
        A = ccs.copy()
        A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)

        bla = np.ones(A.shape[1]).reshape(1,-1)
        bla[0,-1] = 0
        A = np.concatenate((A, bla), axis=0)
        bla = -np.ones(A.shape[1]).reshape(1,-1)
        bla[0,-1] = 0
        A = np.concatenate((A, bla), axis=0)

        for i in range(self.m):
            bla = np.zeros(A.shape[1]).reshape(1,-1)
            bla[0, i] = -1
            A = np.concatenate((A, bla), axis=0)

        b = np.zeros(len(ccs) + 2 + self.m)
        b[len(ccs)] = 1
        b[len(ccs) + 1] = -1
        return A, b

    def corner_weights_new(self):
        ccs = self.ccs.copy()
        ccs = unique_tol(ccs, tol=1e-4)

        import cdd
        def compute_poly_vertices(A, b):
            b = b.reshape((b.shape[0], 1))
            mat = cdd.Matrix(np.hstack([b, -A]), number_type='float')
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1:
                    continue
                if i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        try:
            ccs1 = np.vstack(ccs)
            max_v = np.max(np.abs(ccs1))
            if max_v > 1.0:
                ccs1 = ccs1 / max_v
            ccs1 = np.round(ccs1, decimals=4)
            A, b = self.get_linear_system_corner_weights(ccs1)
            vertices1 = compute_poly_vertices(A, b)
        except:
            try:
                ccs2 = np.vstack(ccs)
                ccs2 = np.round(ccs2, decimals=6)
                A, b = self.get_linear_system_corner_weights(ccs2)
                vertices1 = compute_poly_vertices(A, b)
            except:
                try:
                    ccs3 = np.vstack(ccs)
                    ccs3 = np.round(ccs3, decimals=8)
                    A, b = self.get_linear_system_corner_weights(ccs3)
                    vertices1 = compute_poly_vertices(A, b)
                except:
                    try:
                        ccs4 = np.vstack(ccs)
                        max_v = np.max(np.abs(ccs4))
                        if max_v > 1.0:
                            ccs4 = ccs4 / max_v
                        ccs4 = np.round(ccs4, decimals=6)
                        A, b = self.get_linear_system_corner_weights(ccs4)
                        vertices1 = compute_poly_vertices(A, b)
                    except:
                        print("Failed to compute polyhedron vertices!")
                        return extrema_weights(dim=self.m)

        corner = []
        for v in vertices1:
            cw = v[:-1]
            cw = np.clip(cw, 0.0, 1.0)
            cw = cw / cw.sum()
            corner.append(cw)
        return corner


    def extrema_weights(self) -> List[np.ndarray]:
        extrema_weights = []
        for i in range(self.m):
            w = np.zeros(self.m)
            w[i] = 1.0
            extrema_weights.append(w)
        return extrema_weights

    def is_dominated(self, value: np.ndarray) -> bool:
        if len(self.ccs) == 0: 
            return False
        for w in self.W:
            if np.dot(value, w) > self.max_scalarized_value(w):
                return False
        return True
