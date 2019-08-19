from scipy import optimize
import numpy as np
from utils.floorplan_utils.floorplan_misc import mat_to_cost_map
import cv2
from scipy.misc import imsave
import pdb


def visualize_edges(edges, im_size):
    img = np.zeros([im_size, im_size, 3], dtype=np.uint8)
    for idx, edge in enumerate(edges):
        e1 = (int(np.round(edge[0][0])), int(np.round(edge[0][1])))
        e2 = (int(np.round(edge[1][0])), int(np.round(edge[1][1])))
        cv2.line(img, e1, e2, (255, 0, 0), 1)
        if idx == 0:
            cv2.putText(img, '{}'.format(idx), e1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
    imsave('./all_edges.png', img)


def global_room_cost(x, optimize_idx, initial_edges, edge_rooms, corner_cost_maps, pixel_cost_maps):
    # todo: define the global cost for floorplan sub-graphs

    edge_shifts = np.zeros(len(initial_edges))
    edge_shifts[optimize_idx] = x

    # print('var {}'.format(edge_shifts))
    shifted_edges = list()
    for edge_i, edge in enumerate(initial_edges):
        shifted_edge = shift_edge(edge, edge_shifts[edge_i])
        shifted_edges.append(shifted_edge)

    # fixme: Currently, we don't consider how shifting one edge would influence the other edges, because the
    # fixme: movement is going to be quite small

    ###
    # visulization for debugging purpose
    ###
    # visualize_edges(shifted_edges, im_size=256)
    ###
    # END visulization for debugging purpose
    ###

    edge_constant = 0.1
    intra_cost = 0
    # todo: compute intra-room term, using every room's corner map and pixel-wise cost

    for idx, edge in enumerate(shifted_edges):
        room_idx = edge_rooms[idx]
        corner_map = corner_cost_maps[room_idx]
        pixel_map = pixel_cost_maps[room_idx]
        # because the coordinates are real numbers here, we need interpolation to estimate the cost values
        c_1 = edge[0]
        c_2 = edge[1]
        # Use bilinear interpolation to compute corner, edge cost for real-valued coordinates
        corner_cost = intra_corner_cost(c_1, corner_map)
        edge_cost = intra_edge_cost(c_1, c_2, pixel_map) + edge_constant

        intra_cost += (corner_cost + edge_cost)

    # todo: compute inter-room term based on edge pixel penalty

    im_size = corner_cost_maps[0].shape[0]
    all_pixels = list()
    for y in range(im_size):
        for x in range(im_size):
            all_pixels.append([x, y])
    all_pixels = np.array(all_pixels)
    # inter_cost should be ((im_size*im_size), 1)
    inter_cost = _inter_cost(all_pixels, shifted_edges)

    # visualization
    # cost_map = np.reshape(inter_cost, [im_size, im_size])
    # imsave('inter_cost.png', cost_map)

    inter_cost = np.sum(inter_cost)

    lambda_intra = 1
    lambda_inter = 1
    global_cost = intra_cost * lambda_intra + inter_cost * lambda_inter
    # print('cost {}'.format(global_cost))
    return global_cost


def shift_edge(edge, shift):
    edge_vec = (edge[1][0] - edge[0][0], -(edge[1][1] - edge[0][1]))
    length = np.sqrt(edge_vec[0] ** 2 + edge_vec[1] ** 2)
    norm_vec = (-edge_vec[1] / length, edge_vec[0] / length)
    shift_vec = (norm_vec[0] * shift, norm_vec[1] * shift)
    end_1 = (edge[0][0] + shift_vec[0], edge[0][1] + shift_vec[1])
    end_2 = (edge[1][0] + shift_vec[0], edge[1][1] + shift_vec[1])
    shifted_edge = (end_1, end_2)
    return shifted_edge


def _inter_cost(all_pixels, edges):
    """
        Given a pixel at (x, y), compute the inter-room cost term,
        For every edge, (x, y) will be assigned a value based on the distance to the edge,
        then the final pixel cost will be the maximum cost over all edges.
        This cost encourages close edges to be pulled towards each other while also considering other cost terms
    """
    all_edge_costs = list()
    for edge in edges:
        len_e = np.sqrt((edge[1][1] - edge[0][1]) ** 2 + (edge[1][0] - edge[0][0]) ** 2)
        # 01 means line segment from edge[0] to edge[1]
        # p represents the point (x, y)
        unit_v_01 = ((edge[1][0] - edge[0][0]) / len_e, (edge[1][1] - edge[0][1]) / len_e)
        unit_v_10 = ((edge[0][0] - edge[1][0]) / len_e, (edge[0][1] - edge[1][1]) / len_e)
        len_0p = np.sqrt((all_pixels[:, 0] - edge[0][0]) ** 2 + (all_pixels[:, 1] - edge[0][1]) ** 2)
        len_1p = np.sqrt((all_pixels[:, 0] - edge[1][0]) ** 2 + (all_pixels[:, 1] - edge[1][1]) ** 2)

        dist = np.zeros(all_pixels.shape[0])

        dist[np.where(len_0p == 0)] = 0
        dist[np.where(len_1p == 0)] = 0

        v_0p = (all_pixels[:, 0] - edge[0][0], all_pixels[:, 1] - edge[0][1])
        v_1p = (all_pixels[:, 0] - edge[1][0], all_pixels[:, 1] - edge[1][1])
        cos_p01 = (v_0p[0] * unit_v_01[0] + v_0p[1] * unit_v_01[1]) / len_0p
        cos_p10 = (v_1p[0] * unit_v_10[0] + v_1p[1] * unit_v_10[1]) / len_1p

        # compute the distance from (x, y) to edge
        len_proj_p01 = len_0p * cos_p01
        proj_p01 = (unit_v_01[0] * len_proj_p01, unit_v_01[1] * len_proj_p01)
        norm_p = (edge[0][0] + proj_p01[0], edge[0][1] + proj_p01[1])
        dist_to_edge = np.sqrt((all_pixels[:, 0] - norm_p[0]) ** 2 + (all_pixels[:, 1] - norm_p[1]) ** 2)

        dist[np.where((cos_p01 > 0) & (cos_p10 > 0) & (len_0p != 0) & (len_1p != 0))] = dist_to_edge[
            np.where((cos_p01 > 0) & (cos_p10 > 0) & (len_0p != 0) & (len_1p != 0))]
        dist[np.where((cos_p01 > 0) & (cos_p10 <= 0) & (len_0p != 0) & (len_1p != 0))] = len_1p[
            np.where((cos_p01 > 0) & (cos_p10 <= 0) & (len_0p != 0) & (len_1p != 0))]
        dist[np.where((cos_p10 > 0) & (cos_p01 <= 0) & (len_0p != 0) & (len_1p != 0))] = len_0p[
            np.where((cos_p10 > 0) & (cos_p01 <= 0) & (len_0p != 0) & (len_1p != 0))]

        cost = np.zeros(dist.shape)
        cost[np.where(dist > 3)] = 0
        cost[np.where(dist <= 3)] = np.exp(-dist ** 2 / 3)[np.where(dist <= 3)]
        all_edge_costs.append(cost)

    all_cost = np.stack(all_edge_costs, axis=0)
    inter_cost = np.max(all_cost, axis=0)

    assert (not np.isnan(inter_cost).any()) and (not np.isinf(inter_cost).any())

    return inter_cost


def intra_corner_cost(corner, corner_map):
    if corner[0] - int(corner[0]) == 0 and corner[1] - int(corner[1]) == 0:  # integer coordinates
        val = corner_map[int(corner[1]), int(corner[0])]
    elif corner[0] - int(corner[0]) == 0:
        ceil_y = int(np.ceil(corner[1]))
        floor_y = int(np.floor(corner[1]))
        val = (ceil_y - corner[1]) * corner_map[floor_y, int(corner[0])] + (corner[1] - floor_y) * corner_map[
            ceil_y, int(corner[0])]
    elif corner[1] - int(corner[1]) == 0:
        ceil_x = int(np.ceil(corner[0]))
        floor_x = int(np.floor(corner[0]))
        val = (ceil_x - corner[0]) * corner_map[int(corner[1]), floor_x] + (corner[0] - floor_x) * corner_map[
            int(corner[1]), ceil_x]
    else:
        # use bilinear interpolation to estimate the value
        ceil_y = int(np.ceil(corner[1]))
        floor_y = int(np.floor(corner[1]))
        ceil_x = int(np.ceil(corner[0]))
        floor_x = int(np.floor(corner[0]))
        val = bilinear_interpolation(corner[0], corner[1], [(floor_x, floor_y, corner_map[floor_x, floor_y]),
                                                            (floor_x, ceil_y, corner_map[floor_x, ceil_y]),
                                                            (ceil_x, floor_y, corner_map[ceil_x, floor_y]),
                                                            (ceil_x, ceil_y, corner_map[ceil_x, ceil_y])])
    cost = (1 - val) / 10.0
    return cost


def intra_edge_cost(c1, c2, cost_map):
    if c1[0] - int(c1[0]) == 0 and c1[1] - int(c1[1]) == 0:  # integer coordinates
        cost = cost_map[(int(c1[0]), int(c1[1]))][(int(c2[0]), int(c2[1]))]
    elif c1[0] - int(c1[0]) == 0:
        ceil_y = int(np.ceil(c1[1]))
        floor_y = int(np.floor(c1[1]))
        p_floor = (c1[0], floor_y)
        p_ceil = (c1[0], ceil_y)
        cost_floor = _find_cost(p_floor, c1, c2, cost_map)
        cost_ceil = _find_cost(p_ceil, c1, c2, cost_map)
        cost = (ceil_y - c1[1]) * cost_floor + (c1[1] - floor_y) * cost_ceil
    elif c1[1] - int(c1[1]) == 0:
        ceil_x = int(np.ceil(c1[0]))
        floor_x = int(np.floor(c1[0]))
        p_floor = (floor_x, c1[1])
        p_ceil = (ceil_x, c1[1])
        cost_floor = _find_cost(p_floor, c1, c2, cost_map)
        cost_ceil = _find_cost(p_ceil, c1, c2, cost_map)
        cost = (ceil_x - c1[0]) * cost_floor + (c1[0] - floor_x) * cost_ceil
    else:
        # use bilinear interpolation to estimate the value
        ceil_y = int(np.ceil(c1[1]))
        floor_y = int(np.floor(c1[1]))
        ceil_x = int(np.ceil(c1[0]))
        floor_x = int(np.floor(c1[0]))

        p_ff = (floor_x, floor_y)
        p_fc = (floor_x, ceil_y)
        p_cf = (ceil_x, floor_y)
        p_cc = (ceil_x, ceil_y)

        cost_ff = _find_cost(p_ff, c1, c2, cost_map)
        cost_fc = _find_cost(p_fc, c1, c2, cost_map)
        cost_cf = _find_cost(p_cf, c1, c2, cost_map)
        cost_cc = _find_cost(p_cc, c1, c2, cost_map)

        cost = bilinear_interpolation(c1[0], c1[1], [(floor_x, floor_y, cost_ff),
                                                     (floor_x, ceil_y, cost_fc),
                                                     (ceil_x, floor_y, cost_cf),
                                                     (ceil_x, ceil_y, cost_cc)])

    return cost


def _find_cost(p1, e1, e2, cost_map):
    """
        Given original edge endpoints e1 and e2, and the new endpoint p (shifted from e1), find a best shifted edge,
        and then return its cost
    """
    vec = (e2[0] - e1[0], e2[1] - e1[1])
    p2 = (p1[0] + vec[0], p1[1] + vec[1])

    if p2 in cost_map[p1]:
        cost = cost_map[p1][p2]
    else:
        # todo: find the closest point px to p2, so that cost_map[p1][px] exists
        print('could not find a proper p2')
        pdb.set_trace()
        cost = 0

    return cost


def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)  # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1) + 0.0)


def optimize_floorplan(recon_info):
    """

    :param recon_info: The initial reconstruction information for every room
    :return: new reconstruction results optimized with the global cost function
    """
    dominant_directions = recon_info['dominant_directions']
    room_recons = recon_info['room_recons']

    all_edges = list()
    edge_room_idx = list()
    room_corner_maps = list()
    room_pixel_costs = list()

    for room_idx, room_recon in enumerate(room_recons):
        room_path = room_recon['room_path']
        corner_map = room_recon['corner_heatmap']
        cost_map_mat = room_recon['cost_map_mat']
        cost_map = mat_to_cost_map(cost_map_mat, im_size=corner_map.shape[0])
        room_corner_maps.append(corner_map)
        room_pixel_costs.append(cost_map)
        room_edges = list()
        for node_i, room_node in enumerate(room_path):
            next_node_i = node_i + 1 if node_i < len(room_path) - 1 else 0
            next_node = room_path[next_node_i]
            room_edges.append((room_node, next_node))
        all_edges += room_edges
        edge_room_idx += [room_idx] * len(room_edges)

    # edge_shifts = np.zeros(len(all_edges))

    # edge_shifts = np.random.rand(len(all_edges))
    # cost = global_room_cost(x, all_edges, edge_room_idx, room_corner_maps, room_pixel_costs)

    # pdb.set_trace()

    np.warnings.filterwarnings('ignore')

    # for opt_idx in range(len(all_edges)):
    #     x = 0.0
    #     results = optimize.minimize(global_room_cost, x, args=(opt_idx, all_edges, edge_room_idx, room_corner_maps, room_pixel_costs), options={'maxiter':100})
    #     optimal_shift = results.x
    #     print('move edge {} by {}'.format(opt_idx, optimal_shift))
    #     shifted_edge = shift_edge(all_edges[opt_idx], optimal_shift)
    #     int_shifted_edge = (int(np.round(shifted_edge[0][0])), int(np.round(shifted_edge[0][1]))) , (int(np.round(shifted_edge[1][0])), int(np.round(shifted_edge[1][1])))
    #     all_edges[opt_idx] = int_shifted_edge

    shifted_edge = shift_edge(all_edges[1], 1)
    int_shifted_edge = (int(np.round(shifted_edge[0][0])), int(np.round(shifted_edge[0][1]))) , (int(np.round(shifted_edge[1][0])), int(np.round(shifted_edge[1][1])))
    all_edges[1] = int_shifted_edge

    shifted_edge = shift_edge(all_edges[33], -1)
    int_shifted_edge = (int(np.round(shifted_edge[0][0])), int(np.round(shifted_edge[0][1]))), (
    int(np.round(shifted_edge[1][0])), int(np.round(shifted_edge[1][1])))
    all_edges[33] = int_shifted_edge

    shifted_edge = shift_edge(all_edges[34], 1)
    int_shifted_edge = (int(np.round(shifted_edge[0][0])), int(np.round(shifted_edge[0][1]))), (
    int(np.round(shifted_edge[1][0])), int(np.round(shifted_edge[1][1])))
    all_edges[34] = int_shifted_edge

    pdb.set_trace()
