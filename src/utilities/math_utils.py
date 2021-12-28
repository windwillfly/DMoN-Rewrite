import math

import numpy as np
from typing import Tuple, List


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dims = [self.x, self.y]

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __getitem__(self, item):
        return self.dims[item]

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __array__(self):
        return np.array(self.dims)

    def __repr__(self):
        return f'({self.x}, {self.y})'


class Triangle:
    def __init__(self, p1: Point, p2: Point, p3: Point):
        if self.lineside(p1, p2, p3):
            self.p1 = p1
            self.p2 = p2
            self.p3 = p3
        else:
            self.p1 = p3
            self.p2 = p2
            self.p3 = p1

        self.points = [self.p1, self.p2, self.p3]

    def __repr__(self):
        return f'{self.p1} - {self.p2} - {self.p3}'

    @staticmethod
    def lineside(p1: Point, p2: Point, p3: Point):
        return (p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)

    def __getitem__(self, item):
        return self.points[item]

    def __iter__(self):
        return iter([self.p1, self.p2, self.p3])

    def __array__(self):
        return np.asarray([self.p1.dims, self.p2.dims, self.p3.dims])


class Polygon:

    def __init__(self, points: List[Point]):
        self.x = [p.x for p in points]
        self.y = [p.y for p in points]

    def area(self):
        return 0.5 * np.abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))


class Circle:
    def __init__(self, x, y, r):
        self.center = Point(x, y)
        self.r = r

    def position(self):
        return [self.center.x, self.center.y]

    def distance(self, other):
        return self.center.distance(other.center)

    def is_overlapping(self, other):
        return self.distance(other) <= (self.r + other.r)


def calc_distance(p1: Tuple, p2: Tuple):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def push_touching_nodes(node_pos: dict):
    nodes_as_circles = {k: Circle(*v, 0.03) for k, v in node_pos.items()}

    for first_node_idx, first_node in nodes_as_circles.items():
        for second_node_idx, second_node in nodes_as_circles.items():
            if first_node is second_node:
                continue

            if first_node.is_overlapping(second_node):
                f_distance = first_node.distance(second_node)
                f_overlap = 0.5 * (f_distance - first_node.r - second_node.r)

                disp_x = f_overlap * (first_node.center.x - second_node.center.x) / f_distance
                disp_y = f_overlap * (first_node.center.y - second_node.center.y) / f_distance

                first_node.center.x -= disp_x
                first_node.center.y -= disp_y
                second_node.center.x += disp_x
                second_node.center.y += disp_y

    return {k: v.position() for k, v in nodes_as_circles.items()}


def sutherland_hodgman_on_triangles(first_triangle: Triangle, second_triangle: Triangle):
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return Point(*[(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3])

    outputList = first_triangle
    cp1 = second_triangle[-1]

    for clipVertex in second_triangle:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return outputList
    return outputList


def calc_frustum(person_features, use_body=True, frustum_length=1, frustum_angle=math.pi / 4):
    person_pos = Point(*person_features[:2])
    if use_body:
        head_pose = person_features[2]
    else:
        head_pose = person_features[2] + person_features[3]

    head_pose_left = head_pose + frustum_angle
    head_pose_right = head_pose - frustum_angle

    tri_left_point_x = math.cos(head_pose_left)
    tri_left_point_y = math.sin(head_pose_left)
    tri_left_point = frustum_length * Point(tri_left_point_x,
                                            tri_left_point_y) + person_pos

    tri_right_point_x = math.cos(head_pose_right)
    tri_right_point_y = math.sin(head_pose_right)
    tri_right_point = frustum_length * Point(tri_right_point_x,
                                             tri_right_point_y) + person_pos

    return Triangle(tri_left_point, person_pos, tri_right_point)


if __name__ == '__main__':
    poly1 = Triangle(Point(0, 0), Point(1, 1), Point(-1, 1))
    poly2 = Triangle(Point(2, 2), Point(3, 2), Point(2, 3))
    result = Polygon(sutherland_hodgman_on_triangles(poly1, poly2))

    print(result.area())
