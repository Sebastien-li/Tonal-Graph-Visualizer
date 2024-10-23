""" This module contains the class NoteGraph."""
import partitura as pt
import numpy as np

from src.music_theory_classes import Pitch

class Graph:
    """
    Class to represent a graph with numpy arrays.
    """
    def __init__(self, nodes=None, edge_index=None, edge_attr=None):
        self.nodes = nodes
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    def __getitem__(self, i):
        return self.nodes[i]

    def __len__(self):
        return len(self.nodes)

    def order(self):
        """ Return the number of nodes and edges."""
        return len(self.nodes), len(self.edge_index)

    def get_edge_source(self, src):
        """ Return the index of the edges that have src as source."""
        return np.where(self.edge_index['src'] == src)[0]

    def get_edge_dest(self, dst):
        """ Return the index of the edges that have dst as destination."""
        return np.where(self.edge_index['dst'] == dst)[0]

    def edge_index_to_adj(self):
        """ Return the adjacency matrix of the graph."""
        n = len(self.nodes)
        adj = np.zeros((n,n))
        for edge in self.edge_index:
            adj[edge[0], edge[1]] = 1
        return adj


class NoteGraph(Graph):
    """ Class to represent a graph of notes."""
    dtype_nodes = [ ('id',int),
                    ('pitch_chromatic', int),
                    ('pitch_name', 'U10'),
                    ('pitch_diatonic', int),
                    ('pitch_space', np.float32),
                    ('pitch_octave', int),
                    ('onset', int),
                    ('duration', int),
                    ('offset', int),
                    ('beat', int),
                    ('voice', int)]

    dtype_edges = [('type', 'U10')]

    def __init__(self, score_path):
        super().__init__()
        self.score = pt.load_score(score_path)
        self.create_graph()

    def create_graph(self, find_leap = False):
        """ Create nodes and edges. Optionally, find note leaps. """
        self.nodes = self.create_nodes()
        self.edge_index, self.edge_attr  = self.create_edges()
        if find_leap:
            self.find_leap()

    def create_nodes(self):
        """ Create nodes """
        note_array = pt.utils.music.ensure_notearray(self.score,
                                                     include_time_signature=True,
                                                     include_pitch_spelling=True)
        nodes = np.zeros(len(note_array), dtype=self.dtype_nodes)
        for i, note in enumerate(note_array):
            pitch = Pitch.from_step_alter(note['step'], note['alter'])
            nodes[i]['pitch_chromatic'] = pitch.chromatic
            nodes[i]['pitch_name'] = pitch.name
            nodes[i]['pitch_diatonic'] = pitch.diatonic
            nodes[i]['pitch_space'] = note['pitch']
            nodes[i]['pitch_octave'] = note['octave']
            nodes[i]['onset'] = note['onset_div']
            nodes[i]['duration'] = note['duration_div']
            nodes[i]['offset'] = note['onset_div'] + note['duration_div']
            nodes[i]['beat'] = 1 + note['onset_beat'] % note['ts_beats']
            nodes[i]['voice'] = note['voice']

        nodes = np.sort(np.array(nodes, dtype = self.dtype_nodes), order = ['onset', 'pitch_space'])
        nodes['id'] = np.arange(len(nodes))
        return nodes

    def create_edges(self):
        """ Create edges """
        nodes = self.nodes
        edge_index = list()
        edge_attr = list()
        for i, node in enumerate(nodes):
            for j in np.where(nodes['onset'] == node['onset'])[0]:
                if i < j:
                    edge_index.append((i, j))
                    edge_attr.append(('onset'))

            for j in np.where(nodes['onset'] == node['offset'])[0]:
                if i < j:
                    edge_index.append((i, j))
                    edge_attr.append(('follow'))

            for j in np.where(
                (node['onset'] < nodes['onset']) & (node['offset'] > nodes['onset']))[0]:
                if i < j:
                    edge_index.append((i, j))
                    edge_attr.append(('during'))

        for offset in np.sort(np.unique(nodes['offset']))[:-1]:
            if offset not in nodes['onset']:
                src = np.where(nodes['offset'] == offset)[0]
                diffs = nodes['onset'] - offset
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == np.min(tmp))[0]
                for i in src:
                    for j in dst:
                        edge_index.append((i, j))
                        edge_attr.append(('rest'))

        edge_index = np.array(edge_index, dtype=[('src', int), ('dst', int)])
        edge_attr = np.array(edge_attr, dtype=self.dtype_edges)
        return edge_index, edge_attr

    def get_vertical_dict(self):
        """Vertical dict : onset -> nodes at onset sorted by pitch_space."""
        nodes = self.nodes
        edge_index = self.edge_index
        edge_attr = self.edge_attr

        vertical_dict = {}
        for node in nodes:
            onset = node['onset']
            if onset not in vertical_dict:
                vertical_dict[onset] = []
                for inc_edge_idx in self.get_edge_dest(node['id']):
                    inc_node = edge_index[inc_edge_idx]['src']
                    inc_edge_attr = edge_attr[inc_edge_idx]
                    if inc_edge_attr['type'] in ['during']:
                        vertical_dict[onset].append(nodes[nodes['id']==inc_node][0])

            insert_index = np.searchsorted([x['pitch_space'] for x in vertical_dict[onset]],
                                           node['pitch_space'])
            vertical_dict[onset].insert(insert_index, node)

        for onset, nodes in vertical_dict.items():
            vertical_dict[onset] = np.array(nodes, dtype = nodes[0].dtype)

        return vertical_dict

    def find_leap(self):
        """ Find the leap nodes."""
        is_leap = []
        for node in self.nodes:
            if node['isRest']:
                is_leap.append(False)
                continue

            inc_edg_idx = self.get_edge_dest(node['id'])
            out_edg_idx = self.get_edge_source(node['id'])
            inc_edg_idx = inc_edg_idx[self.edge_attr[inc_edg_idx]['type'] != 'onset']
            out_edg_idx = out_edg_idx[self.edge_attr[out_edg_idx]['type'] != 'onset']
            inc_nodes = self[self.edge_index[inc_edg_idx]['src']]
            out_nodes = self[self.edge_index[out_edg_idx]['dst']]
            inc_edges_attr = self.edge_attr[inc_edg_idx]
            out_edges_attr = self.edge_attr[out_edg_idx]

            inc_nodes = inc_nodes[np.logical_and(inc_edges_attr['type'] != 'onset',
                                                 ~inc_nodes['isRest'])]

            out_nodes = out_nodes[np.logical_and(out_edges_attr['type'] != 'onset',
                                                 ~out_nodes['isRest'])]

            if inc_nodes.size==0:
                is_leap.append(False)
                continue

            if out_edg_idx.size==0:
                is_leap.append(False)
                continue

            closest_inc_index = np.argmin([abs(x['pitch_space']-node['pitch_space'])
                                            for x in inc_nodes])
            inc_node = inc_nodes[closest_inc_index]
            prev_interval = abs(7*node['pitch_octave']+node['pitch_diatonic'] - 7*inc_node['pitch_octave']-inc_node['pitch_diatonic'])

            closest_out_index = np.argmin([abs(x['pitch_space']-node['pitch_space'])
                                            for x in out_nodes])
            out_node = out_nodes[closest_out_index]
            next_interval = abs(7*out_node['pitch_octave']+out_node['pitch_diatonic'] - 7*node['pitch_octave']-node['pitch_diatonic'])

            is_leap.append(prev_interval > 1 and next_interval > 1)

        new_dtype = self.dtype_nodes + [('isLeap', bool)]
        nodes = np.zeros_like(self.nodes, dtype = new_dtype)
        for name,_ in self.dtype_nodes:
            nodes[name] = self.nodes[name]
        nodes['isLeap'] = is_leap
        self.nodes = nodes
