import numpy as np
from collections import defaultdict
from utils.data_utils import compute_bin_idx, get_edge_pixels
import pdb


def build_room_pixel_graph(all_nodes, directions_rad, starting_line, inter_region, inter_edge, inter_corner, mask, room_edge_map,
                           corner_heatmap, room_bbox, density_map):
    graph = dict()
    for node_idx in range(len(all_nodes)):
        graph[node_idx] = dict()
    node_indices = list(range(len(all_nodes)))

    cost_map, intra_map = _pre_compute_cost_map(room_bbox, inter_region, inter_edge, mask, room_edge_map, corner_heatmap,
                                     directions_rad, density_map)
    complexity_term = 1
    for pixel_node, pixel_node_idx in zip(all_nodes, node_indices):
        for other_node, other_idx in zip(all_nodes, node_indices):
            if pixel_node_idx == other_idx or pixel_node_idx in graph and other_idx in graph[pixel_node_idx]:
                continue
            if other_idx in graph and pixel_node_idx in graph[other_idx]:  # symmetric
                graph[pixel_node_idx][other_idx] = graph[other_idx][pixel_node_idx]
            if not (pixel_node in cost_map and other_node in cost_map[pixel_node]):  # todo: Using dominant directions
                continue
            if _check_intersect(pixel_node, other_node, starting_line[0], starting_line[1]):
                continue
            edge_cost = cost_map[pixel_node][other_node] + complexity_term  # the constant for every edge
            if mask[other_node[1], other_node[0]] != 1:  # encourage path to go through global corners
                # HIGH corner conf --> 0 cost, low -> at most 0.1
                corner_cost = 0.2 * (1 - corner_heatmap[other_node[1], other_node[0]]) + 0.2 * (1 - inter_corner[other_node[1], other_node[0]])
                # corner_cost = 0.2 * (1 - corner_heatmap[other_node[1], other_node[0]])
                # corner_cost = 0
                edge_cost += corner_cost
            graph[pixel_node_idx][other_idx] = edge_cost
            if not (other_idx in graph and pixel_node_idx in graph[other_idx]):
                graph[other_idx][pixel_node_idx] = graph[pixel_node_idx][other_idx]
        # print('finish processning node {}'.format(pixel_node))

    return graph, all_nodes, cost_map, starting_line, intra_map


def _check_intersect(A, B, C, D):
    o1 = _ccw(A, B, C)
    o2 = _ccw(A, B, D)
    o3 = _ccw(C, D, A)
    o4 = _ccw(C, D, B)

    if o1 != o2 and o3 != o4:
        return True

    if (o1 == 0 and _on_edge(A, C, B)) or (o2 == 0 and _on_edge(A, D, B)) or (o3 == 0 and _on_edge(C, A, D)) or (
            o4 == 0 and _on_edge(C, B, D)):
        return True

    return False


def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _on_edge(p, q, r):
    """
        p, q, r are **known to be colinear**  This function Checks  whether q lies on p-r
    """
    if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
        return True
    return False


def _pre_compute_cost_map(bbox, inter_region, inter_edge, mask, edgeness, corner_heatmap, directions, density_img):
    cost_map = defaultdict(dict)
    intra_map = defaultdict(dict)

    for direction in directions:
        _cost_on_direction(cost_map, bbox, inter_region, inter_edge, mask, edgeness, corner_heatmap, direction, density_img, intra_map)

    return cost_map, intra_map


def _cost_on_direction(cost_map, bbox, inter_region, inter_edge, mask, edgeness, corner_heatmap, direction, density_img, intra_map):
    assert 0 <= direction < np.pi
    # note that we need to swap y-axis to simplify the computation
    # we build up the coordinate system according to the bbox
    if direction < np.pi / 2:
        diag_end_1 = (0, bbox[3])
        diag_end_2 = (bbox[2], 0)
        diag_type = 'main'
    else:
        diag_end_1 = (0, 0)
        diag_end_2 = (bbox[2], bbox[3])
        diag_type = 'anti'
    diag_pixels = get_edge_pixels(diag_end_1, diag_end_2)
    # O(N^2) for one direction

    for diag_pixel in diag_pixels:
        p1, p2 = _get_direction_endpoints(diag_pixel, direction, bbox, diag_type)
        direction_nodes = get_edge_pixels(p1, p2)
        for node_i in range(1, len(direction_nodes)):
            local_node = tuple(direction_nodes[node_i])
            global_node = (local_node[0] + bbox[0], (bbox[3] - local_node[1]) + bbox[1])
            node_x, node_y = global_node
            local_cost = 0

            intra_local_cost = 0

            if mask[node_y, node_x] == 1:  # intra-room region cost
                local_cost += 100

            edgeness_cost = (1 - edgeness[node_y, node_x]) * 0.2 + (1-inter_edge[node_y, node_x]) * 0.1  # intra-room edge cost + inter-room edge cost
            intra_edgeness_cost = (1 - edgeness[node_y, node_x]) * 0.2
            # edgeness_cost = (1 - density_img[node_y, node_x]) * 0.2

            local_cost = local_cost + edgeness_cost

            intra_local_cost = intra_local_cost + intra_edgeness_cost
            for to_node_i in range(node_i - 1, -1, -1):
                local_to_node = tuple(direction_nodes[to_node_i])
                global_to_node = (bbox[0] + local_to_node[0], bbox[1] + (bbox[3] - local_to_node[1]))
                if node_i - to_node_i == 1:
                    memorized = 0
                    memorized_intra = 0
                else:
                    global_prev_node = (
                        direction_nodes[node_i - 1][0] + bbox[0], (bbox[3] - direction_nodes[node_i - 1][1]) + bbox[1])
                    memorized = cost_map[global_to_node][global_prev_node]
                    memorized_intra = intra_map[global_to_node][global_prev_node]
                cost_map[global_node][global_to_node] = local_cost + memorized
                cost_map[global_to_node][global_node] = local_cost + memorized

                intra_map[global_node][global_to_node] = intra_local_cost + memorized_intra
                intra_map[global_to_node][global_node] = intra_local_cost + memorized_intra

    return cost_map, intra_map


def _get_direction_endpoints(diag_point, direction, bbox, diag_type):
    tangent = np.tan(direction)
    xk, yk = diag_point
    if diag_type == 'main':
        assert 0 <= direction < np.pi / 2
        if direction == 0:
            p1 = (bbox[2], yk)
            p2 = (0, yk)
        else:
            y1 = bbox[3]
            x1 = int(np.round((y1 - yk) / tangent + xk))
            x2 = bbox[2]
            y2 = int(np.round((x2 - xk) * tangent + yk))
            if x1 <= bbox[2]:
                p1 = (x1, y1)
            else:
                assert y2 <= bbox[3]
                p1 = (x2, y2)
            y1 = 0
            x1 = int(np.round((y1 - yk) / tangent + xk))
            x2 = 0
            y2 = int(np.round((x2 - xk) * tangent + yk))
            if x1 >= 0:
                p2 = (x1, y1)
            else:
                assert y2 >= 0
                p2 = (x2, y2)
    elif diag_type == 'anti':
        assert np.pi / 2 <= direction < np.pi
        if direction == np.pi / 2:
            p1 = (xk, bbox[3])
            p2 = (xk, 0)
        else:
            y1 = bbox[3]
            x1 = int(np.round((y1 - yk) / tangent + xk))
            x2 = 0
            y2 = int(np.round((x2 - xk) * tangent + yk))
            if x1 >= 0:
                p1 = (x1, y1)
            else:
                assert y2 <= bbox[3]
                p1 = (x2, y2)
            y1 = 0
            x1 = int(np.round((y1 - yk) / tangent + xk))
            x2 = bbox[2]
            y2 = int(np.round((x2 - xk) * tangent + yk))
            if x1 <= bbox[2]:
                p2 = (x1, y1)
            else:
                assert y2 >= 0
                p2 = (x2, y2)
    else:
        raise ValueError('Invalid diagonal type{}'.format(diag_type))

    return p1, p2


if __name__ == '__main__':
    diag_point = (25, 50)
    direction = np.pi * 0.75
    bbox = [30, 30, 50, 100]
    diag_type = 'anti'
    p1, p2 = _get_direction_endpoints(diag_point, direction, bbox, diag_type)
    pdb.set_trace()
