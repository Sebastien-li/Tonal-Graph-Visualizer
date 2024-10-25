from ..mei_svg import MEISVGHandler
import plotly.graph_objects as go

import numpy as np

color_palette = {'red': 'orangered', 'orange' : 'darkorange',
                 'yellow' : 'gold',
                 'blue': 'royalblue', 'light_blue':'skyblue',
                 'green': 'forestgreen', 'light_green': 'limegreen',
                 'white': 'white', 'black': 'black', 'transparent_white' : 'rgba(255,255,255,0.2)',
                  'gray': 'gray' }

color_dict = {'onset': color_palette['red'],
              'during': color_palette['blue'],
              'follow': color_palette['green'],
              'silence': color_palette['light_green']}

def plot_score(url, note_graph):
    """ Plot the score of the piece. """
    mei_svg_doc = MEISVGHandler.parse_url(url)

    fig = go.Figure()
    graph_scale = .8
    graph_width = mei_svg_doc.svg_width * graph_scale
    graph_height = mei_svg_doc.svg_height * graph_scale


    fig.add_layout_image(x=0, sizex=graph_width, y=0, sizey=graph_height, xref="x", yref="y",
                         layer="below",
                         source = url)
    fig.update_xaxes(showgrid=False, visible=False, range=[0,graph_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[graph_height,0])
    fig.update_layout(height=graph_height, width=graph_width, margin={'l':0, 'r':0, 'b':0, 't':0},
                      plot_bgcolor='white')
    for i, edge_index in enumerate(note_graph.edge_index):
        edge_attr = note_graph.edge_attr[i]['type']
        src = note_graph.nodes[edge_index[0]]
        dst = note_graph.nodes[edge_index[1]]
        src_x, src_y = mei_svg_doc.get_note_coords(src['note_id'], (graph_width, graph_height))
        dst_x, dst_y = mei_svg_doc.get_note_coords(dst['note_id'], (graph_width, graph_height))
        trace = go.Scatter(
            x = [src_x, dst_x, None],
            y = [src_y, dst_y, None],
            line = {"color": color_dict[edge_attr]},
            hoverinfo='none',
            mode='lines',
            opacity=0.5,
            showlegend=False,
        )
        fig.add_trace(trace)
    return fig