import math
from itertools import combinations

import csv
import networkx as nx
import os
from utilities.math_utils import calc_distance, calc_frustum, sutherland_hodgman_on_triangles, Polygon


class BaseConverter:

    def __init__(self):
        pass

    @staticmethod
    def _read_csv(csv_file):
        with open(csv_file) as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                yield row


class SalsaConverter(BaseConverter):

    def __init__(self, root_folder, edges_from_gt=True):
        super().__init__()
        self.salsa_root = root_folder
        self.show_ground_truth = edges_from_gt
        self.fformation_csv = os.path.join(self.salsa_root, 'fformationGT.csv')
        self.geometry_root = os.path.join(self.salsa_root, 'geometryGT')
        self.unit_scale = 1
        self.max_dist = 6
        if 'CMU' in self.salsa_root:
            self.unit_scale = 100

    def _parse_fformation(self, fformation_csv):
        fformation = {}
        timestamps = []
        for csv_line in self._read_csv(fformation_csv):
            timestamp = float(csv_line[0])
            if 'CMU' in self.salsa_root:
                group = [int(person.strip()) + 1 for person in csv_line[1:] if person != '']
            else:
                group = [int(person.strip()) for person in csv_line[1:] if person != '']
            if timestamp not in fformation:
                timestamps.append(timestamp)
                fformation[timestamp] = [group]
            elif group in fformation[timestamp]:
                continue
            else:
                fformation[timestamp].append(group)

        return fformation, timestamps

    def _parse_geometry(self, geometry_root):
        geometry = {}
        timestamps = []
        for geo_file in os.listdir(geometry_root):
            person_no = int(os.path.splitext(os.path.basename(geo_file))[0])
            if 'CMU' in self.salsa_root:
                person_no += 1
            person_geometry = {}
            for csv_line in self._read_csv(os.path.join(geometry_root, geo_file)):
                timestamp, pos_x, pos_y, _, body_pose, rel_head_pose, valid = list(map(float, csv_line))
                pos_x /= self.unit_scale
                pos_y /= self.unit_scale
                if person_no == 1:
                    timestamps.append(timestamp)
                valid = bool(valid)
                if timestamp not in person_geometry:
                    person_geometry[timestamp] = [pos_x, pos_y, body_pose, rel_head_pose]
                else:
                    person_geometry[timestamp].append(pos_x, pos_y, body_pose, rel_head_pose)

            geometry[person_no] = person_geometry

        return geometry, timestamps

    def convert(self, edge_distance_threshold=0.4, frustum_length: float = 1.0, frustum_angle=math.pi / 3, ):
        frustum_max_area = frustum_length * math.tan(frustum_angle)
        fformation, ff_timestamps = self._parse_fformation(self.fformation_csv)
        geometry, geo_timestamps = self._parse_geometry(self.geometry_root)

        graphs = []

        for cur_ts in ff_timestamps:
            cur_ff = fformation[cur_ts]

            graph = nx.Graph()
            node_list = []
            edge_list = []

            colors = ['#38761d', '#01579b', '#fb8c00', '#e77865', '#cbeaad', '#6180c3', '#69de4b', '#c72792', '#6d2827',
                      '#1e2157', '#58C0CF', '#167C54', '#B76E09', '#265A98', '#AE45ED', '#98900B', '#85D54B']

            edge_nodes = []
            edge_weights = []
            extra_groups = 0

            # For each person in geometry dict
            for person_no in range(1, len(geometry) + 1):
                for group_no, group_details in enumerate(cur_ff):

                    # Check if he's in group
                    if person_no in group_details:

                        try:
                            person_feat = geometry[person_no][cur_ts]
                        except KeyError:
                            continue
                        # Add him to nodes with correct membership
                        node_list.append((person_no,
                                          {'membership': group_no, 'color': colors[group_no], 'feats': person_feat,
                                           'person_no': person_no, 'ts': cur_ts}))
                        break
                else:
                    # If he is not in a group, meaning he is not labeled.
                    # We must add him to node list, with new membership
                    try:
                        person_feat = geometry[person_no][cur_ts]
                    except KeyError:
                        # If he is not in scene, do not add him to the node_list
                        continue
                    group_no = len(cur_ff) + extra_groups
                    extra_groups += 1
                    node_list.append((person_no,
                                      {'membership': group_no, 'color': colors[group_no], 'feats': person_feat,
                                       'person_no': person_no, 'ts': cur_ts}))

            # Show edges from ground truth labels
            if self.show_ground_truth:
                for group_details in cur_ff:
                    for p1, p2 in combinations(group_details, 2):
                        edge_list.append((p1, p2, {'weight': 1}))
            else:
                # Else, construct graph with weighted edges
                # Inversely correlated with distance between two pairs of people
                for person_no in range(1, len(geometry) + 1):
                    for other_person_no in range(person_no + 1, len(geometry) + 1):
                        try:
                            person_feat = geometry[person_no][cur_ts]
                        except KeyError:
                            continue
                        person_loc = person_feat[:2]

                        try:
                            other_person_feat = geometry[other_person_no][cur_ts]
                        except KeyError:
                            continue
                        other_person_loc = other_person_feat[:2]

                        dist = calc_distance(person_loc, other_person_loc)
                        edge_nodes.append((person_no, other_person_no))

                        f1 = calc_frustum(person_feat, frustum_length=frustum_length, frustum_angle=frustum_angle)
                        f2 = calc_frustum(other_person_feat, frustum_length=frustum_length, frustum_angle=frustum_angle)

                        intersection_poly = Polygon(sutherland_hodgman_on_triangles(f1, f2))
                        weight = math.exp(-dist / (2 * (self.max_dist ** 2)))
                        frustum_multiplier = intersection_poly.area() / (frustum_max_area / 4)
                        weight *= frustum_multiplier

                        edge_weights.append(weight)

                # Normalize distances btw 0-1. -> Subtract norm_dist from 1 to get edge weight.
                # norm_edge_weigths = 1 - (edge_weights / np.max(edge_weights))
                for (p1, p2), weight in zip(edge_nodes, edge_weights):

                    # Weed out edges above a certain threshold
                    if weight > edge_distance_threshold:
                        edge_list.append((p1, p2, {'weight': weight}))

            group_numbers = list({node[1]['membership'] + 1 for node in node_list})
            total_groups = len(group_numbers)
            missing_group = total_groups * (total_groups + 1) // 2 - sum(group_numbers)
            if missing_group != 0:
                missing_group = (total_groups + 1) * (total_groups + 2) // 2 - sum(group_numbers)
                for node in node_list:
                    if node[1]['membership'] >= missing_group:
                        node[1]['membership'] -= 1

            graph.add_nodes_from(node_list)
            graph.add_edges_from(edge_list)
            graphs.append(graph)

        return graphs


if __name__ == '__main__':
    from utilities.visualization import show_gt_graph, show_results_on_graph

    frustum_angle = math.pi / 4
    frustum_length = 1
    edge_distance_threshold = 0.15
    sc = SalsaConverter(root_folder=os.path.join('data', 'salsa_ps_fold1', 'train'), edges_from_gt=False)
    graphs = sc.convert(frustum_angle=frustum_angle, frustum_length=frustum_length,
                        edge_distance_threshold=edge_distance_threshold)

    graph_num = 128
    current_graph = graphs[graph_num]

    # current_graph.remove_nodes_from([node[0] for node in current_graph.nodes(data=True) if node[1]['membership'] != 4])
    show_results_on_graph(current_graph, frame_no=str(graph_num), save_path='.', title="Salsa Poster Session",
                          draw_frustum=True, frustum_angle=frustum_angle,
                          frustum_length=frustum_length)
    #show_gt_graph(current_graph, title="Salsa Poster Session", draw_frustum=True, frustum_angle=frustum_angle,
    #              frustum_length=frustum_length)

    # with open(r'test_experiments\10\latent_graphs.json') as f:
    #     latent_positions = json.load(f)
    #
    # epoch_positions = []
    # for epoch_num, latent_pos_of_graphs in latent_positions.items():
    #     epoch_positions.append(latent_pos_of_graphs[graph_num])
    #
    # for i in range(0, len(latent_positions), 50):
    #     cur_graph = current_graph.copy()
    #     current_positions = epoch_positions[i]
    #     node_features = {node_num: {'feats': [current_positions[node_num][0], current_positions[node_num][1], 0, 0]} for
    #                      node_num in range(cur_graph.number_of_nodes())}
    #     nx.set_node_attributes(cur_graph, {'feats': []})
    #     show_gt_graph(cur_graph, f'94_epoch_{i}', draw_frustum=False)
