# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize module utils."""


import numpy as np
import networkx as nx
import imageio
from matplotlib import pylab as pl
import re
import os
import glob
from past.builtins import xrange
from future.utils import iteritems

pos = None
G = None


def readFile():
    """Read the file and return the indices as list of lists."""
    filename = 's.txt'
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]
    return array2d


def read_file_draw_graph():
    """Create the graph and returns the networkx version of it 'G'."""
    global pos
    global G
    array2d = readFile()

    ROW, COLUMN = len(array2d), len(array2d[0])
    count = 0

    G = nx.Graph()

    for j in xrange(COLUMN):
        for i in xrange(ROW):
            if array2d[ROW - 1 - i][j] == 0:
                G.add_node(count, pos=(j, i))
                count += 1

    pos = nx.get_node_attributes(G, 'pos')

    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] - 1:
                G.add_edge(index, index2, weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] - 1:
                G.add_edge(index, index2, weight=1)

    return G


class GNG():
    """."""

    def __init__(self, data, eps_b=0.05, eps_n=0.0005, max_age=25,
                 lambda_=100, alpha=0.5, d=0.0005, max_nodes=100):
        """."""
        self.graph = nx.Graph()
        self.data = data
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_age = max_age
        self.lambda_ = lambda_
        self.alpha = alpha
        self.d = d
        self.max_nodes = max_nodes
        self.num_of_input_signals = 0

        self.pos = None

        node1 = data[np.random.randint(0, len(data))]
        node2 = data[np.random.randint(0, len(data))]

        # make sure you dont select same positions
        if node1[0] == node2[0] and node1[1] == node2[1]:
            print("Rerun ---------------> similar nodes selected")
            return None

        # initialize here
        self.count = 0
        self.graph.add_node(self.count, pos=(node1[0], node1[1]), error=0)
        self.count += 1
        self.graph.add_node(self.count, pos=(node2[0], node2[1]), error=0)
        self.graph.add_edge(self.count - 1, self.count, age=0)

    def distance(self, a, b):
        """Calculate distance between two points."""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def determine_2closest_vertices(self, curnode):
        """Where this curnode is actually the x,y index of the data we want to analyze."""
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        templist = []
        for node, position in iteritems(self.pos):
            dist = self.distance(curnode, position)
            templist.append([node, dist])

        distlist = np.array(templist)

        ind = np.lexsort((distlist[:, 0], distlist[:, 1]))
        distlist = distlist[ind]

        return distlist[0], distlist[1]

    def get_new_position(self, winnerpos, nodepos):
        """."""
        move_delta = [self.eps_b * (nodepos[0] - winnerpos[0]), self.eps_b * (nodepos[1] - winnerpos[1])]
        newpos = [winnerpos[0] + move_delta[0], winnerpos[1] + move_delta[1]]

        return newpos

    def get_new_position_neighbors(self, neighborpos, nodepos):
        """."""
        movement = [self.eps_n * (nodepos[0] - neighborpos[0]), self.eps_n * (nodepos[1] - neighborpos[1])]
        newpos = [neighborpos[0] + movement[0], neighborpos[1] + movement[1]]

        return newpos

    def update_winner(self, curnode):
        """."""
        # find nearest unit and second nearest unit
        winner1, winner2 = self.determine_2closest_vertices(curnode)
        winnernode = winner1[0]
        winnernode2 = winner2[0]
        win_dist_from_node = winner1[1]

        errorvectors = nx.get_node_attributes(self.graph, 'error')

        error1 = errorvectors[winner1[0]]
        # update the new error
        newerror = error1 + win_dist_from_node**2
        self.graph.add_node(winnernode, error=newerror)

        # move the winner node towards current node
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        newposition = self.get_new_position(self.pos[winnernode], curnode)
        self.graph.add_node(winnernode, pos=newposition)

        # now update all the neighbors distances and their ages
        neighbors = nx.all_neighbors(self.graph, winnernode)
        age_of_edges = nx.get_edge_attributes(self.graph, 'age')
        for n in neighbors:
            newposition = self.get_new_position_neighbors(self.pos[n], curnode)
            self.graph.add_node(n, pos=newposition)
            key = (int(winnernode), n)
            if key in age_of_edges:
                newage = age_of_edges[(int(winnernode), n)] + 1
            else:
                newage = age_of_edges[(n, int(winnernode))] + 1
            self.graph.add_edge(winnernode, n, age=newage)

        # no sense in what I am writing here, but with algorithm it goes perfect
        # if winnner and 2nd winner are connected, update their age to zero
        if (self.graph.get_edge_data(winnernode, winnernode2) is not None):
            self.graph.add_edge(winnernode, winnernode2, age=0)
        else:
            # else create an edge between them
            self.graph.add_edge(winnernode, winnernode2, age=0)

        # if there are ages more than maximum allowed age, remove them
        age_of_edges = nx.get_edge_attributes(self.graph, 'age')
        for edge, age in iteritems(age_of_edges):

            if age > self.max_age:
                self.graph.remove_edge(edge[0], edge[1])

                # if it causes isolated vertix, remove that vertex as well

                for node in self.graph.nodes():
                    if not self.graph.neighbors(node):
                        self.graph.remove_node(node)

    def get_average_dist(self, a, b):
        """."""
        av_dist = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

        return av_dist

    def save_img(self, fignum, output_images_dir='images'):
        """."""
        fig = pl.figure(fignum)
        ax = fig.add_subplot(111)

        nx.draw(G, pos, node_color='#ffffff', with_labels=False, node_size=100, alpha=0.5, width=1.5)

        position = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, position, node_color='r', node_size=100, with_labels=False, edge_color='b', width=1.5)
        pl.title('Growing Neural Gas')
        pl.savefig("{0}/{1}.png".format(output_images_dir, str(fignum)))

        pl.clf()
        pl.close(fignum)

    def train(self, max_iterations=10000, output_images_dir='images'):
        """."""

        if not os.path.isdir(output_images_dir):
            os.makedirs(output_images_dir)

        print("Ouput images will be saved in: {0}".format(output_images_dir))
        fignum = 0
        self.save_img(fignum, output_images_dir)

        for i in xrange(1, max_iterations):
            print("Iterating..{0:d}/{1}".format(i, max_iterations))
            for x in self.data:
                self.update_winner(x)

                # step 8: if number of input signals generated so far
                if i % self.lambda_ == 0 and len(self.graph.nodes()) <= self.max_nodes:
                    # find a node with the largest error
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    import operator
                    node_largest_error = max(iteritems(errorvectors), key=operator.itemgetter(1))[0]

                    # find a node from neighbor of the node just found,
                    # with largest error
                    neighbors = self.graph.neighbors(node_largest_error)
                    max_error_neighbor = None
                    max_error = -1
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    for n in neighbors:
                        if errorvectors[n] > max_error:
                            max_error = errorvectors[n]
                            max_error_neighbor = n

                    # insert a new unit half way between these two
                    self.pos = nx.get_node_attributes(self.graph, 'pos')

                    newnodepos = self.get_average_dist(self.pos[node_largest_error], self.pos[max_error_neighbor])
                    self.count = self.count + 1
                    newnode = self.count
                    self.graph.add_node(newnode, pos=newnodepos)

                    # insert edges between new node and other two nodes
                    self.graph.add_edge(newnode, max_error_neighbor, age=0)
                    self.graph.add_edge(newnode, node_largest_error, age=0)

                    # remove edge between the other two nodes

                    self.graph.remove_edge(max_error_neighbor, node_largest_error)

                    # decrease error variable of other two nodes by multiplying with alpha
                    errorvectors = nx.get_node_attributes(self.graph, 'error')
                    error_max_node = self.alpha * errorvectors[node_largest_error]
                    error_max_second = self.alpha * max_error
                    self.graph.add_node(max_error_neighbor, error=error_max_second)
                    self.graph.add_node(node_largest_error, error=error_max_node)

                    # initialize the error variable of newnode with max_node
                    self.graph.add_node(newnode, error=error_max_node)

                    fignum += 1
                    self.save_img(fignum, output_images_dir)

                # step 9: Decrease all error variables
                errorvectors = nx.get_node_attributes(self.graph, 'error')
                for i in self.graph.nodes():
                    olderror = errorvectors[i]
                    newerror = olderror - self.d * olderror
                    self.graph.add_node(i, error=newerror)


def main():
    """."""
    global pos, G
    G = read_file_draw_graph()

    inList = []
    for key, value in iteritems(pos):
        inList.append([value[0], value[1]])

    mat = np.array(inList, dtype='float64')
    return mat


def sort_nicely(limages):
    """."""
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    limages = sorted(limages, key=alphanum_key)
    return limages


def convert_images_to_gif(output_images_dir, output_gif):
    """Convert a list of images to a gif."""

    image_dir = "{0}/*.png".format(output_images_dir)
    list_images = glob.glob(image_dir)
    file_names = sort_nicely(list_images)
    images = [imageio.imread(fn) for fn in file_names]
    imageio.mimsave(output_gif, images)


if __name__ == "__main__":

    data = main()
    grng = GNG(data)
    output_images_dir = 'images'
    output_gif = "output.gif"
    if grng is not None:
        grng.train(max_iterations=10000)
        convert_images_to_gif(output_images_dir, output_gif)
