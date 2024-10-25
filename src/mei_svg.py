from xml.dom import minidom

class MEISVGHandler():

    def __init__(self, doc):
        self.doc = doc
        self.svg_width, self.svg_height, self.scaled_width, self.scaled_height, self.translator_x, self.translator_y = self.get_size()

    @classmethod
    def parse_url(cls, url):
        doc = minidom.parse(url)
        return cls(doc)

    @classmethod
    def parse_string(cls, data):
        doc = minidom.parseString(data)
        return cls(doc)

    def get_size(self):
        svg_width = scaled_viewbox = self.doc.getElementsByTagName('svg')[0].getAttribute('width')
        svg_width = int(svg_width.split('px')[0])
        svg_height = scaled_viewbox = self.doc.getElementsByTagName('svg')[0].getAttribute('height')
        svg_height = int(svg_height.split('px')[0])
        scaled_viewbox = self.doc.getElementsByTagName('svg')[1].getAttribute('viewBox')
        scaled_viewbox = scaled_viewbox.split(' ')
        scaled_width = int(scaled_viewbox[2])
        scaled_height = int(scaled_viewbox[3])
        translator = [x for x in self.doc.getElementsByTagName('g') if x.getAttribute('class')=='page-margin'][0]
        translator = translator.getAttribute('transform').split('(')[1].split(')')[0].split(',')
        translator_x = int(translator[0])
        translator_y = int(translator[1])
        return svg_width, svg_height, scaled_width, scaled_height, translator_x, translator_y

    def xy_scaler(self, coord, graph_size, note_width = 720):
        x, y = coord
        graph_width, graph_height = graph_size
        new_x = (x+self.translator_x+note_width/5) * (graph_width / self.scaled_width)
        new_y = (y+self.translator_y) * (graph_height / self.scaled_height)
        return new_x, new_y

    def get_note_coords(self, note_id, graph_size):
        note_element = [x for x in self.doc.getElementsByTagName('g') if x.getAttribute('id') == note_id][0]
        note_head_element = note_element.getElementsByTagName('use')[0]
        x = int(note_head_element.getAttribute('x'))
        y = int(note_head_element.getAttribute('y'))
        width = note_head_element.getAttribute('width')
        width = int(width.split('px')[0])
        return self.xy_scaler((x,y), graph_size, width)