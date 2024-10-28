""" Module to handle the parsing of SVG files generated by Verovio """
from xml.dom import minidom

class MEISVGHandler():
    """ Class to handle the parsing of SVG files generated by Verovio """

    def __init__(self, doc):
        self.doc = doc
        returned = self.get_size()
        self.svg_width = returned[0]
        self.svg_height = returned[1]
        self.scaled_width = returned[2]
        self.scaled_height = returned[3]
        self.translator_x = returned[4]
        self.translator_y = returned[5]
        self.note_element_dict = self.compute_note_element_dict()

    @classmethod
    def parse_url(cls, url):
        """ Parse an SVG file from a URL """
        doc = minidom.parse(url)
        return cls(doc)

    @classmethod
    def parse_string(cls, data):
        """ Parse an SVG file from a string """
        doc = minidom.parseString(data)
        return cls(doc)

    def get_size(self):
        """ Get all the size information from the SVG file """
        svg_width = self.doc.getElementsByTagName('svg')[0].getAttribute('width')
        svg_width = int(svg_width.split('px')[0])
        svg_height = self.doc.getElementsByTagName('svg')[0].getAttribute('height')
        svg_height = int(svg_height.split('px')[0])
        scaled_viewbox = self.doc.getElementsByTagName('svg')[1].getAttribute('viewBox')
        scaled_viewbox = scaled_viewbox.split(' ')
        scaled_width = int(scaled_viewbox[2])
        scaled_height = int(scaled_viewbox[3])
        g_tags = self.doc.getElementsByTagName('g')
        translator = [x for x in g_tags if x.getAttribute('class') == 'page-margin'][0]
        translator = translator.getAttribute('transform').split('(')[1].split(')')[0].split(',')
        translator_x = int(translator[0])
        translator_y = int(translator[1])
        return svg_width, svg_height, scaled_width, scaled_height, translator_x, translator_y

    def xy_scaler(self, coord, graph_size, note_width = 720, edge_type = 'start'):
        """ Convert the coordinates from the SVG file to the graph coordinates """
        x, y = coord
        graph_width, graph_height = graph_size
        if edge_type == 'start':
            x_offset = 0
        elif edge_type == 'end':
            x_offset = 0.30 * note_width
        else:
            x_offset = 0.15 * note_width
        new_x = (x+self.translator_x+x_offset) * (graph_width / self.scaled_width)
        new_y = (y+self.translator_y) * (graph_height / self.scaled_height)
        return new_x, new_y

    def get_note_coords(self, note_id, graph_size, edge_type = 'start'):
        """ Get the coordinates of a note from its ID"""
        assert edge_type in ['start','end','middle']
        g_tags = self.doc.getElementsByTagName('g')
        note_head_element = self.note_element_dict[note_id][0]
        x = int(note_head_element.getAttribute('x'))
        y = int(note_head_element.getAttribute('y'))
        width = note_head_element.getAttribute('width')
        width = int(width.split('px')[0])
        return self.xy_scaler((x,y), graph_size, width, edge_type)

    def compute_note_element_dict(self):
        """ dict of element_id: note_element (only for notes)"""
        g_tags = self.doc.getElementsByTagName('g')
        note_element_dict = {x.getAttribute('id'):x.getElementsByTagName('use') for x in g_tags}
        return note_element_dict