import _init_paths
import numpy as np
from collections import defaultdict
from utils.floorplan_utils.floorplan_misc import get_intersection, len_edge, unit_v_edge
from utils.floorplan_utils.graph_algorithms import check_on_edge_approx
from utils.data_utils import draw_room_seg_from_edges

import pdb


def merge_room_graphs(all_room_edges):
    all_room_edges, removed_indices = filter_rooms(all_room_edges, im_size=256)

    all_colinear_pairs = find_all_colinear_paris(all_room_edges)

    colinear_sets = combine_colinear_edges(all_colinear_pairs)

    for colinear_set in colinear_sets:
        edges_to_merge = list(colinear_set)
        edges_to_merge = sorted(edges_to_merge, key=lambda x: -x[0])
        merged_edges = merge_edges(edges_to_merge)

        for merged_edge, old_edge in zip(merged_edges, edges_to_merge):
            if len(merged_edge) > 0:
                assert merged_edge[0][0] == old_edge[0]
                room_idx = merged_edge[0][0]
                
                if len(merged_edge) == 1 and merged_edge[0] == old_edge:
                    continue
                # change room graph accordingly
                replaced_idx = all_room_edges[room_idx].index(old_edge[1])
                all_room_edges[room_idx].pop(replaced_idx)
                for new_idx, new_edge in enumerate(merged_edge):
                    insert_idx = new_idx + replaced_idx
                    all_room_edges[room_idx].insert(insert_idx, new_edge[1])
            else:
                room_idx = old_edge[0]
                replaced_idx = all_room_edges[room_idx].index(old_edge[1])
                all_room_edges[room_idx].pop(replaced_idx)

    # take intersection for every rooms to recover the room structure
    refined_room_edges = [adjust_room_edges(room_edges) for room_edges in all_room_edges]

    # clean every room loop by removing I-shape corners
    cleaned_room_edges = clean_room_edges(refined_room_edges)

    global_graph = defaultdict(list)
    for room_edges in cleaned_room_edges:
        for edge in room_edges:
            c1, c2 = edge
            global_graph[c1] += [c2, ]
            global_graph[c2] += [c1, ]

    for corner in global_graph:
        global_graph[corner] = set(global_graph[corner])

    all_room_masks = list()
    all_room_paths = list()
    for room_edges in cleaned_room_edges:
        room_mask, _, _ = draw_room_seg_from_edges(room_edges, 256)
        all_room_masks.append(room_mask)
        all_room_paths.append(_extract_room_path(room_edges))

    return global_graph, cleaned_room_edges, all_room_masks, all_room_paths, removed_indices


def filter_rooms(all_room_edges, im_size):
    # filter rooms that are covered by larger rooms
    room_masks = list()
    for room_edges in all_room_edges:
        room_mask, _, _ = draw_room_seg_from_edges(room_edges, im_size)
        room_masks.append(room_mask)

    removed = list()
    for room_idx, room_mask in enumerate(room_masks):
        # do not consider the current room, and do not consider removed rooms
        other_masks = [room_masks[i] for i in range(len(all_room_edges)) if i != room_idx and i not in removed]
        if len(other_masks) == 0:  # if all other masks are removed..
            other_masks_all = np.zeros([im_size, im_size])
        else:
            other_masks_all = np.clip(np.sum(np.stack(other_masks, axis=-1), axis=-1), 0, 1)
        joint_mask = np.clip(other_masks_all + room_mask, 0, 1)
        mask_area = room_mask.sum()
        overlap_area = mask_area + other_masks_all.sum() - joint_mask.sum()
        if overlap_area / mask_area > 0.5:
            removed.append(room_idx)

    all_room_edges = [all_room_edges[idx] for idx in range(len(all_room_edges)) if idx not in removed]

    return all_room_edges, removed


def clean_room_edges(all_room_edges):
    refined_room_paths = [_extract_room_path(room_edges) for room_edges in all_room_edges]
    corner_to_room = defaultdict(list)
    for room_idx, room_path in enumerate(refined_room_paths):
        for corner in room_path:
            corner_to_room[corner].append(room_idx)
    # remove I-shape corner used by only one room
    for room_idx, room_edges in enumerate(all_room_edges):
        cp_room_edges = list(room_edges)
        rm_flag = True
        while rm_flag:
            rm_flag = False
            for edge_i, edge in enumerate(cp_room_edges):
                prev_i = edge_i - 1
                prev_edge = cp_room_edges[prev_i]
                if _check_colinear(prev_edge, edge):
                    rm_candidate = edge[0]
                    if len(corner_to_room[rm_candidate]) == 1 and corner_to_room[rm_candidate][0] == room_idx:
                        cp_room_edges[prev_i] = (prev_edge[0], edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                        break
                next_i = edge_i + 1 if edge_i < len(cp_room_edges) - 1 else 0
                next_edge = cp_room_edges[next_i]
                if _check_colinear(next_edge, edge):
                    rm_candidate = edge[1]
                    if len(corner_to_room[rm_candidate]) == 1 and corner_to_room[rm_candidate][0] == room_idx:
                        cp_room_edges[next_i] = (edge[0], next_edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                        break
        if len(cp_room_edges) != len(room_edges):
            all_room_edges[room_idx] = cp_room_edges

    corner_to_room = get_corner_to_room(all_room_edges)
    all_corners = list(corner_to_room.keys())
    corners_to_merge = find_corners_to_merge(all_corners)
    while corners_to_merge is not None:
        num_aff = [len(corner_to_room[x]) for x in corners_to_merge]
        order = np.argsort(num_aff)[::-1]
        base_corner = corners_to_merge[order[0]]
        for corner in corners_to_merge:
            if corner == base_corner:
                continue
            all_room_edges = move_corner(corner, base_corner, corner_to_room, all_room_edges)

        corner_to_room = get_corner_to_room(all_room_edges)
        all_corners = list(corner_to_room.keys())
        corners_to_merge = find_corners_to_merge(all_corners)

    for room_idx, room_edges in enumerate(all_room_edges):
        cp_room_edges = list(room_edges)
        rm_flag = True
        while rm_flag:
            rm_flag = False
            for edge_i, edge in enumerate(cp_room_edges):
                len_e = len_edge(edge)
                if len_e <= 5:
                    if len(corner_to_room[edge[0]]) == 1:
                        prev_i = edge_i - 1
                        prev_edge = cp_room_edges[prev_i]
                        cp_room_edges[prev_i] = (prev_edge[0], edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                        break
                    elif len(corner_to_room[edge[1]]) == 1:
                        next_i = edge_i + 1 if edge_i < len(cp_room_edges) - 1 else 0
                        next_edge = cp_room_edges[next_i]
                        cp_room_edges[next_i] = (edge[0], next_edge[1])
                        rm_flag = True
                        cp_room_edges.pop(edge_i)
                    else:
                        continue

        if len(cp_room_edges) != len(room_edges):
            all_room_edges[room_idx] = cp_room_edges

    return all_room_edges


def get_corner_to_room(all_room_edges):
    room_paths = [_extract_room_path(room_edges) for room_edges in all_room_edges]
    corner_to_room = defaultdict(list)
    for room_idx, room_path in enumerate(room_paths):
        for corner in room_path:
            corner_to_room[corner].append(room_idx)
    return corner_to_room


def move_corner(c, target, corner_to_room, all_room_edges):
    rooms = corner_to_room[c]
    for room_idx in rooms:
        for edge_idx, edge in enumerate(all_room_edges[room_idx]):
            if c in edge:
                if c == edge[0]:
                    new_edge = (target, edge[1])
                elif c == edge[1]:
                    new_edge = (edge[0], target)
                else:
                    continue
                all_room_edges[room_idx][edge_idx] = new_edge
    return all_room_edges





def find_corners_to_merge(all_corners):
    all_close_pairs = list()
    for idx1, corner in enumerate(all_corners):
        for idx2, other_corner in enumerate(all_corners):
            if idx2 <= idx1:
                continue
            if len_edge((corner, other_corner)) <= 3:
                all_close_pairs.append([corner, other_corner])

    if len(all_close_pairs) == 0:
        return None

    close_set = find_one_close_set(all_close_pairs)
    corners_to_merge = list(close_set)

    return corners_to_merge


def _extract_room_path(room_edges):
    room_path = [edge[0] for edge in room_edges]
    return room_path


def find_one_close_set(all_corner_paris):
    all_pairs = list(all_corner_paris)  # make a copy of the input list
    combined = [False] * len(all_corner_paris)

    close_set = _combine_colinear_pairs(0, all_pairs, combined)

    return close_set


def find_all_colinear_paris(all_room_edges):
    colinear_pairs = list()
    for room_idx, room_edges in enumerate(all_room_edges):
        for edge_idx, edge in enumerate(room_edges):
            for other_room_idx, other_edges in enumerate(all_room_edges):
                if other_room_idx < room_idx:
                    continue
                for other_edge_idx, other_edge in enumerate(other_edges):
                    if other_room_idx == room_idx and other_edge_idx <= edge_idx:
                        continue
                    if _check_colinear(edge, other_edge):
                        ele1 = (room_idx, edge)
                        ele2 = (other_room_idx, other_edge)
                        colinear_pairs.append([ele1, ele2])
    return colinear_pairs


def _refine_room_path(room_path):
    if check_on_edge_approx(room_path[1], room_path[0], room_path[-1]):
        room_path = room_path[1:]
    if check_on_edge_approx(room_path[-2], room_path[0], room_path[-1]):
        room_path = room_path[:-1]
    edge_source_sink = (room_path[-1], room_path[0])
    if _check_colinear(edge_source_sink, (room_path[0], room_path[1])):
        room_path = room_path[1:]
    if _check_colinear(edge_source_sink, (room_path[-2], room_path[-1])):
        room_path = room_path[:-1]
    
    updated = True
    # keep removing too short edges to simplify the final visualization
    while updated:
        room_edges = [(node, room_path[node_i+1]) for node_i, node in enumerate(room_path[:-1])]
        room_edges.append((room_path[-1], room_path[0]))
        updated = False
        for edge_i, edge in enumerate(room_edges):
            len_e = len_edge(edge)
            if len_e <= 5:
                next_i = edge_i + 1 if edge_i < len(room_edges) - 1 else 0
                prev_i = edge_i - 1
                next_edge = room_edges[next_i]
                unit_next = unit_v_edge(next_edge)
                ext_next = ((next_edge[0][0] - unit_next[0] * 10, next_edge[0][1] - unit_next[1] * 10), (next_edge[1][0] + unit_next[0] * 10, next_edge[1][1] + unit_next[1] * 10))
                prev_edge = room_edges[prev_i]
                unit_prev = unit_v_edge(prev_edge)
                ext_prev = ((prev_edge[0][0] - unit_prev[0] * 10, prev_edge[0][1] - unit_prev[1] * 10), (prev_edge[1][0] + unit_prev[0] * 10, prev_edge[1][1] + unit_prev[1] * 10))
                intersect = get_intersection(ext_prev[0], ext_prev[1], ext_next[0], ext_next[1])
                if intersect is None:
                    continue
                else:
                    updated = True
                    intersect = (int(np.round(intersect[0])), int(np.round(intersect[1])))
                    room_edges[prev_i] = (prev_edge[0], intersect)
                    room_edges[next_i] = (intersect, next_edge[1])
                    room_edges.pop(edge_i)
                    room_path = _extract_room_path(room_edges)
                    break

    return room_path


def adjust_room_edges(room_edges):
    refined_room_edges = list()
    # first filter collasped edges
    
    for edge_i, edge in enumerate(room_edges):
        next_i = edge_i
        while True:
            next_i = next_i + 1 if next_i < len(room_edges) - 1 else 0
            next_edge = room_edges[next_i]
            if next_edge[0] != next_edge[1]:
                break
        if edge[1] == next_edge[0]:  # no need for refining
            refined_room_edges.append(edge)
        else:  # the two corners disagree, refinement is required
            if edge[0] == edge[1]:
                print('skip collasped edge')
                continue
            unit_edge = unit_v_edge(edge)
            ext_edge = ((edge[0][0] - unit_edge[0] * 10, edge[0][1] - unit_edge[1] * 10), (edge[1][0] + unit_edge[0] * 10, edge[1][1] + unit_edge[1] * 10))
            unit_next = unit_v_edge(next_edge)
            ext_next = ((next_edge[0][0] - unit_next[0] * 10, next_edge[0][1] - unit_next[1] * 10), (next_edge[1][0] + unit_next[0] * 10, next_edge[1][1] + unit_next[1] * 10))
            intersec = get_intersection(ext_edge[0], ext_edge[1], ext_next[0], ext_next[1])
            try:
                assert intersec is not None
            except:
                # pdb.set_trace()
                print('no intersect, move endpoint directly')
                intersec = next_edge[0]
            intersec = (int(np.round(intersec[0])), int(np.round(intersec[1])))
            refined_edge = (edge[0], intersec)
            refined_room_edges.append(refined_edge)
            room_edges[edge_i] = refined_edge
            room_edges[next_i] = (intersec, next_edge[1])
            if next_i < edge_i:
                refined_room_edges[next_i] = room_edges[next_i]

    # drop collapsed edges
    refined_room_edges = [edge for edge in refined_room_edges if edge[0] != edge[1]]
    for edge_i in range(len(refined_room_edges)):
        next_i = edge_i + 1 if edge_i < len(refined_room_edges) - 1 else 0
        if refined_room_edges[edge_i][1] != refined_room_edges[next_i][0]:
            new_edge = (refined_room_edges[edge_i][0], refined_room_edges[next_i][0])
            refined_room_edges[edge_i] = new_edge
    return refined_room_edges


def merge_edges(edges):
    base_e = edges[0][1]
    merged_edges = [edges[0], ]
    base_len = np.sqrt((base_e[1][0] - base_e[0][0]) ** 2 + (base_e[1][1] - base_e[0][1]) ** 2)
    base_unit_v = ((base_e[1][0] - base_e[0][0]) / base_len, (base_e[1][1] - base_e[0][1]) / base_len)

    for edge in edges[1:]:
        room_idx = edge[0]
        e = edge[1]
        v_b0e0 = (e[0][0] - base_e[0][0], e[0][1] - base_e[0][1])
        proj_len = (v_b0e0[0] * base_unit_v[0] + v_b0e0[1] * base_unit_v[1])
        proj_e0 = (int(base_e[0][0] + base_unit_v[0] * proj_len), int(base_e[0][1] + base_unit_v[1] * proj_len))
        proj_e1 = (int(proj_e0[0] + e[1][0] - e[0][0]), int(proj_e0[1] + e[1][1] - e[0][1]))
        new_e = (proj_e0, proj_e1)
        new_edge = (room_idx, new_e)
        merged_edges.append(new_edge)

    adjusted_merged_edges = adjust_colinear_edges(merged_edges)

    return adjusted_merged_edges


def adjust_colinear_edges(edges):
    base_corner = (edges[0][0], edges[0][1][0])
    all_corners = [base_corner, (edges[0][0], edges[0][1][1])]
    for edge in edges[1:]:
        all_corners.append((edge[0], edge[1][0]))
        all_corners.append((edge[0], edge[1][1]))
    unit_v = unit_v_edge(edges[0][1])
    corner_projs = list()
    # FIXME: need to fix the corner coords here, it's wrong now!! they are not merged...
    for room, other_c in all_corners:
        v_base_c = (other_c[0] - base_corner[1][0], other_c[1] - base_corner[1][1])
        proj = (unit_v[0] * v_base_c[0] + unit_v[1] * v_base_c[1])
        corner_projs.append(proj)
    order = np.argsort(corner_projs).tolist()
    # merge corners that are too close to the prev corner
    for o_idx, corner_idx in enumerate(order[1:]):
        corner = all_corners[corner_idx][1]
        prev_idx = order[o_idx]
        prev_corner = all_corners[prev_idx][1]
        dist = len_edge((corner, prev_corner))
        if dist <= 5:
            all_corners[corner_idx] = (all_corners[corner_idx][0], prev_corner)        

    adjusted_edges = list()
    for idx, edge in enumerate(edges):
        room_idx = edge[0]
        idx_1 = idx * 2
        idx_2 = idx * 2 + 1
        adj_idx_1 = order.index(idx_1)
        adj_idx_2 = order.index(idx_2)
        step_direction = 1 if adj_idx_2 > adj_idx_1 else -1
        adjusted_edge = list()
        for o_idx in range(adj_idx_1, adj_idx_2, step_direction):
            c_idx = order[o_idx]
            next_c_idx = order[o_idx + step_direction]
            segment = (room_idx, (all_corners[c_idx][1], all_corners[next_c_idx][1]))
            if len_edge(segment[1]) == 0:
                continue
            adjusted_edge.append(segment) 
        adjusted_edges.append(adjusted_edge)
    return adjusted_edges



def combine_colinear_edges(colinear_pairs):
    all_colinear_sets = list()
    all_pairs = list(colinear_pairs)  # make a copy of the input list
    combined = [False] * len(colinear_pairs)

    while len(all_pairs) > 0:
        colinear_set = _combine_colinear_pairs(0, all_pairs, combined)
        all_colinear_sets.append(colinear_set)
        all_pairs = [all_pairs[i] for i in range(len(all_pairs)) if combined[i] is False]
        combined = [False] * len(all_pairs)
    return all_colinear_sets


def _combine_colinear_pairs(idx, all_pairs, combined):
    colinear_set = set(all_pairs[idx])
    combined[idx] = True
    for other_idx, pair in enumerate(all_pairs):
        if not combined[other_idx] and (all_pairs[idx][0] in all_pairs[other_idx] or all_pairs[idx][1] in all_pairs[other_idx]):
            colinear_set = colinear_set.union(_combine_colinear_pairs(other_idx, all_pairs, combined))
    return colinear_set


def _check_colinear(e1, e2):
    # first check whether two line segments are parallel to each other, if not, return False directly
    len_e1 = len_edge(e1)
    len_e2 = len_edge(e2)
    # we need to always make e2 the shorter one
    if len_e1 < len_e2:
        e1, e2 = e2, e1
    v1_01 = (e1[1][0] - e1[0][0], e1[1][1] - e1[0][1])
    v1_10 = (e1[0][0] - e1[1][0], e1[0][1] - e1[1][1])
    v2_01 = (e2[1][0] - e2[0][0], e2[1][1] - e2[0][1])
    v2_10 = (e2[0][0] - e2[1][0], e2[0][1] - e2[1][1])
    len_1 = np.sqrt(v1_01[0] ** 2 + v1_01[1] ** 2)
    len_2 = np.sqrt(v2_01[0] ** 2 + v2_01[1] ** 2)
    cos = (v1_01[0] * v2_01[0] + v1_01[1] * v2_01[1]) / (len_1 * len_2)
    if abs(cos) > 0.99:
        # then check the distance between two parallel lines
        len_10_20 = len_edge((e1[0], e2[0]))
        len_10_21 = len_edge((e1[0], e2[1]))
        len_11_20 = len_edge((e1[1], e2[0]))
        len_11_21 = len_edge((e1[1], e2[1]))

        # two endpoints are very close, then we can say these two edges are colinear
        if np.min([len_10_20, len_10_21, len_11_20, len_11_21]) <= 5:
            return True
        # otherwise we need to check the distance first
        v_10_20 = (e2[0][0] - e1[0][0], e2[0][1] - e1[0][1])
        cos_11_10_20 = (v1_01[0] * v_10_20[0] + v1_01[1] * v_10_20[1]) / (len_1 * len_10_20)
        sin_11_10_20 = np.sqrt(1 - cos_11_10_20 ** 2)
        dist_20_e1 = len_10_20 * sin_11_10_20
        if dist_20_e1 <= 5:
            # we need two check whether they have some overlaps
            v_11_20 = (e2[0][0] - e1[1][0], e2[0][1] - e1[1][1])
            cos_10_11_20 = (v1_10[0] * v_11_20[0] + v1_10[1] * v_11_20[1]) / (len_1 * len_11_20)
            if cos_11_10_20 >= 0 and cos_10_11_20 >= 0:
                return True
            v_10_21 = (e2[1][0] - e1[0][0], e2[1][1] - e1[0][1])
            cos_11_10_21 = (v1_01[0] * v_10_21[0] + v1_01[1] * v_10_21[1]) / (len_1 * len_10_21)
            v_11_21 = (e2[1][0] - e1[1][0], e2[1][1] - e1[1][1])
            cos_10_11_21 = (v1_10[0] * v_11_21[0] + v1_10[1] * v_11_21[1]) / (len_1 * len_11_21)
            if cos_11_10_21 >= 0 and cos_10_11_21 >= 0:
                return True
            return False
        else:
            # if the two line segments have distance > 3, we can say they are not colinear
            return False
    else:
        return False

if __name__ == '__main__':
    _check_colinear(((186, 135), (186, 174)), ((186, 183), (186, 132)))
    _check_colinear(((186, 183), (186, 132)), ((186, 135), (186, 174)))
