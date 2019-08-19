"""
    Graph algorithms for solving floorplan structures
"""

import numpy as np


###
#  Dijkstra's algorithm and related sub functions
###


def dijkstra(graph, source_idx, end_idx, all_nodes):
    dists = [np.inf for _ in range(len(graph))]
    prevs = [None for _ in range(len(graph))]
    unvisited = [i for i in range(len(graph))]

    dists[source_idx] = 0

    while len(unvisited) > 0:
        min_node = _find_min_node(dists, unvisited)
        unvisited.remove(min_node)

        for neighbour_node in graph[min_node].keys():
            prev_node_idx = prevs[min_node]
            if neighbour_node == prev_node_idx:
                continue  # skip the incoming path, it's not possible to go back to prev again
            if prev_node_idx is not None:
                if check_path_overlapping(all_nodes[neighbour_node], all_nodes[prev_node_idx],
                                          all_nodes[min_node]) or check_on_edge_approx(all_nodes[neighbour_node],
                                                                                       all_nodes[prev_node_idx],
                                                                                       all_nodes[min_node]):
                    continue  # we do not allow path overlapping
            dist = graph[min_node][neighbour_node]
            if dist + dists[min_node] < dists[neighbour_node]:
                dists[neighbour_node] = dist + dists[min_node]
                prevs[neighbour_node] = min_node

    path = list()
    next_node = end_idx

    while next_node is not None:
        path.append(next_node)
        next_node = prevs[next_node]

    return path, dists


def _find_min_node(dists, unvisited):
    min_dist = None
    min_node = None
    for node in unvisited:
        if min_node is None:
            min_node = node
            min_dist = dists[node]
        elif dists[node] < min_dist:
            min_node = node
            min_dist = dists[node]
    return min_node


###
#  End of Dijkstra's algorithm
###


###
#  DFS based graph search algorithm for finding room structure
###

def dfs_all_paths(graph, room_corners, source_idx, end_idx, trial_num):
    all_paths = list()  # keep record of all paths from source to the end
    all_lens = list()

    _dfs(graph, room_corners, source_idx, end_idx, [source_idx, ], 0, all_paths, all_lens, trial_num)

    return all_paths, all_lens


def _dfs(graph, room_corners, current, end, path, path_len, all_paths, all_lens, trial_num):
    if current == end:  # terminating condition
        all_paths.append(path)
        all_lens.append(float(path_len))
        return
    if path_len > 6 + trial_num * 5:
        return
    elif check_corner_on_path(current, path, room_corners):
        return  # if the corner lies on the path, we skip it to avoid overlapping

    for neighbour_node in graph[current].keys():
        dist = graph[current][neighbour_node]
        if neighbour_node in path or dist == np.inf:
            continue  # avoid loop + skip inf, if the node is in the current path, skip it

        path = path + [neighbour_node, ]
        path_len += dist

        _dfs(graph, room_corners, neighbour_node, end, path, path_len, all_paths, all_lens, trial_num)

        path = path[:-1]  # pop the last node
        path_len -= dist


def check_corner_on_path(corner_idx, path, corners):
    path_corners = [corners[idx] for idx in path]
    corner = corners[corner_idx]

    if len(path) < 3:
        return False

    if check_path_overlapping(corner, path_corners[-3], path_corners[-2]):
        return True

    for idx, end_1 in enumerate(path_corners[:-2]):
        end_2 = path_corners[idx + 1]
        if check_on_edge_approx(corner, end_1, end_2):
            return True

    return False


def check_on_edge_approx(corner, end_1, end_2, thresh=-0.95):
    vec_1 = (end_1[0] - corner[0], end_1[1] - corner[1])
    vec_2 = (end_2[0] - corner[0], end_2[1] - corner[1])

    dist_to_1 = np.sqrt((vec_1[0] ** 2 + vec_1[1] ** 2))
    dist_to_2 = np.sqrt((vec_2[0] ** 2 + vec_2[1] ** 2))
    if dist_to_1 == 0 or dist_to_2 == 0:
        return True
    cos = (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / (dist_to_1 * dist_to_2)
    if cos < thresh:
        return True
    else:
        return False


def check_path_overlapping(corner, end_1, end_2):
    vec_1 = (end_1[0] - corner[0], end_1[1] - corner[1])
    vec_2 = (end_2[0] - corner[0], end_2[1] - corner[1])

    dist_to_1 = np.sqrt((vec_1[0] ** 2 + vec_1[1] ** 2))
    dist_to_2 = np.sqrt((vec_2[0] ** 2 + vec_2[1] ** 2))
    if dist_to_1 == 0 or dist_to_2 == 0:
        import pdb
        pdb.set_trace()
    cos = (vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]) / (dist_to_1 * dist_to_2)
    if cos >= 0.99 and dist_to_1 <= dist_to_2:
        return True
    else:
        return False


def reselect_path(all_paths, all_lens, num_nodes, trial_num):
    if num_nodes >= 10:
        thresh = 0.5
    elif num_nodes >= 6:
        thresh = 1.5
    else:
        thresh = 2.5

    thresh = thresh * (1 + trial_num)

    candidate_indices = list()
    avg_lens = list()

    for idx, (path, total_len) in enumerate(zip(all_paths, all_lens)):
        avg_len = total_len / len(path)
        avg_lens.append(avg_len)
        if avg_len < thresh and len(path) > 0.8 * num_nodes:
            candidate_indices.append(idx)

    if len(candidate_indices) > 0:
        candi_paths = [all_paths[i] for i in candidate_indices]
        candi_avg_lens = np.array([avg_lens[i] for i in candidate_indices])
        selected_idx = np.argmin(candi_avg_lens)
        selected_path = candi_paths[selected_idx]
        return selected_path
    else:
        return None

###
#  END DFS based graph search
###
