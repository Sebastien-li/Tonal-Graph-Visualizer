""" Plotter module for the app. """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.mei_svg import MEISVGHandler
from src.rhythm_tree import RhythmTreeInteractive
from src.music_theory_classes import Pitch
from src.roman_text import RomanNumeral


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

def plot_time_graph(note_graph, rhythm_tree, tonal_graph, xmin=None, xmax=None):
    """ Create a plotly figure with the temporal graph of the analysis. """
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Note graph', 'Tonal graph', 'Rhythm tree'),
                        specs=[[{}, {'rowspan': 2}],
                               [{}, None]])
    fig.update_layout(
        showlegend=True, dragmode='pan',
        height = 900, width = 1900,
        title_text = "Temporal graphs",
        legend_tracegroupgap=110,
    )

    # X axis layout
    score = note_graph.score
    if xmin is None:
        xmin = float(score[0].measures[0].start.t)
    if xmax is None:
        last_measure = score[0].measures[min(1, len(score[0].measures)-1)]
        xmax = float(last_measure.end.t) + 0.2

    fig.update_xaxes(range = [xmin, xmax],
                     tickvals = [x.start.t for x in score[0].measures],
                     ticktext = [x.number for x in score[0].measures],
                     matches = 'x')

    fig.update_layout(xaxis_showticklabels=False,
                      xaxis2_showticklabels=True,xaxis2_title = 'measure number',
                      xaxis3_showticklabels=True)

    #Note graph
    edge_trace, node_traces = make_note_graph_trace(note_graph)
    for trace in edge_trace:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(node_traces, row=1, col=1)

    # Rhythm tree
    rt_fill, rt_selected_fill, rt_border, rt_text = make_rhythm_tree_trace(rhythm_tree)
    for trace in rt_fill:
        fig.add_trace(trace, row=2, col=1)
    for trace in rt_selected_fill:
        fig.add_trace(trace, row=2, col=1)
    for trace in rt_border:
        fig.add_trace(trace, row=2, col=1)
    fig.add_trace(rt_text, row=2, col=1)

    # Tonal graph
    node_trace, chord_text_trace, row_height = make_tonal_graph_trace(tonal_graph)
    fig.add_trace(node_trace, row=1, col=2)
    fig.add_trace(chord_text_trace, row=1, col=2)


    # Y axis layout
    fig.update_yaxes(showgrid = False, row=1, col=1, title='pitch space')
    fig.update_yaxes(showgrid = True,
                     zeroline = False,
                     tickvals = [-0.75,0.25,1.25,2.25],
                     ticktext = ['1/2','1','2','4'],
                     row=2, col=1,
                     title = 'subdivision')
    fig.update_yaxes(zeroline = False,
                tickvals = row_height-0.5,
                ticktext = ['C','C#', 'D','E-', 'E', 'F', 'F#', 'G', 'A-', 'A','B-', 'B', ''],
                row=1, col=2)

    return fig

def make_note_graph_trace(note_graph):
    """ Create the traces for the note graph"""
    edge_traces = []
    edge_show_legend = {'onset':True, 'during':True, 'follow':True, 'silence':True}
    for i,edge in enumerate(note_graph.edge_index):
        u = note_graph.nodes[edge[0]]
        v = note_graph.nodes[edge[1]]
        edge_attr = note_graph.edge_attr[i]['type']
        edge_traces.append(go.Scatter(
            x = [u['onset'], v['onset'], None],
            y = [u['pitch_space'], v['pitch_space'], None],
            line={'color':color_dict[edge_attr]},
            hoverinfo='none',
            mode='lines',
            showlegend=edge_show_legend[edge_attr],
            name = edge_attr,
            legendgroup='note_graph',
        ))
        edge_show_legend[edge_attr] = False

    nodes = note_graph.nodes
    node_traces = go.Scatter(
        x = nodes['onset'],
        y = nodes['pitch_space'],
        hovertext = nodes['pitch_name'],
        mode='markers',
        hoverinfo='text',
        name='Notes',
        marker={
            'color':color_palette['white'],
            'size':10,
            'line_width':2},
        showlegend=False,
    )

    return edge_traces, node_traces

def make_rhythm_tree_trace(rhythm_tree: RhythmTreeInteractive):
    """ Create the traces for the rhythm tree"""
    text_x = []
    text_y = []
    text = []
    rectangle_fill_selected_traces = []
    rectangle_fill_traces = []
    rectangle_border_traces = []
    selected_show_legend = True
    unselected_show_legend = True
    for i, node in enumerate(rhythm_tree.depth_first_search()):
        x0 = node.onset
        x1 = node.offset - 0.1
        y0 = np.log2(float(node.subdivision)) if node.depth != 0 else 3
        y1 = np.log2(float(node.subdivision))+0.5 if node.depth != 0 else 3.5

        if node.selected_chord is None:
            pitch, quality_label = '', ''
        else:
            root_diatonic, root_chromatic, quality_index = node.selected_chord
            pitch = Pitch(root_diatonic, root_chromatic)
            quality = rhythm_tree.qualities[int(quality_index)]
            inversion = node.inversion[root_diatonic, root_chromatic, quality_index]
            quality_label = quality.label_with_inversion(inversion)

        # Rectangle fill
        if node.selected:
            rectangle_fill_selected_traces.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                line={'color':color_palette['orange']},
                hoverinfo='text' ,
                text = f"{pitch}{quality_label}" if node.depth != 0 else "Entire score",
                fillcolor=color_palette['orange'],
                customdata=[i],
                showlegend=selected_show_legend,
                name = "Selected analysis",
                ))
            selected_show_legend = False
        else:
            rectangle_fill_traces.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                line={'color':color_palette['light_blue']},
                hoverinfo='text',
                text = f"{pitch}{quality_label}" if node.depth != 0 else "Entire score",
                fillcolor=color_palette['light_blue'],
                customdata=[i],
                showlegend=unselected_show_legend,
                name = "Analysis",
                ))
            unselected_show_legend = False

        # Rectangle border
        rectangle_border_traces.append(go.Scatter(
            x = [x0,x1,x1,x0,x0,None],
            y = [y0,y0,y1,y1,y0,None],
            mode='lines',
            line={'color':color_palette['gray']},
            hoverinfo='skip' ,
            showlegend=False,
            ))

        # Text

        text_x.append((x0+x1)/2)
        text_y.append((y0+y1)/2)
        text.append(f"{pitch}{quality_label}" if node.depth != 0 else "Entire score")

    text_trace = go.Scatter(
        x=text_x,
        y=text_y,
        text=text,
        mode='text',
        textposition='middle center',
        hoverinfo='skip',
        textfont={
            'size':14,
            'color':'black'
        },
        showlegend=False
        )

    return rectangle_fill_traces,rectangle_fill_selected_traces,rectangle_border_traces,text_trace

def make_tonal_graph_trace(tonal_graph):
    """ Create the traces for the tonal graph"""
    row_height = np.zeros(12)
    for onset in tonal_graph.onsets:
        nodes = tonal_graph.nodes[tonal_graph.nodes['onset'] == onset]
        chromatics, counts = np.unique(nodes['tonic_chromatic'], return_counts=True)
        for chromatic,count in zip(chromatics,counts):
            row_height[chromatic] = max(row_height[chromatic],count)
    row_height += 1
    row_height = np.cumsum(np.concatenate(([0],row_height)))

    node_x = []
    node_y = []
    rn_name = []
    annotations = []
    color = []
    rt_id = []
    for onset in tonal_graph.onsets:
        current_y = np.copy(row_height)
        nodes = tonal_graph.nodes[tonal_graph.nodes['onset'] == onset]
        for node in nodes:
            roman_numeral = RomanNumeral.from_tonal_graph_node(tonal_graph, node['id'])
            x = node['onset']
            y = current_y[node['tonic_chromatic']]
            current_y[node['tonic_chromatic']] += 1
            node_x.append(x)
            node_y.append(y)
            rn_name.append(f"{roman_numeral.full_name_with_key}")
            annotations.append(f"{node['weight']:.3f}")
            color.append(color_palette['red'] if node['selected'] else color_palette['black'])
            rt_id.append(node['rhythm_tree_id'])

    node_trace = go.Scatter(
        x = node_x,
        y = node_y,
        mode='markers',
        text=annotations,
        hoverinfo='text',
        marker = {
            'color' : color,
            'size' : 10,
            'line_width' : 2
        },
        opacity=0,
        showlegend=False,
        customdata=rt_id
        )

    chord_text_trace = go.Scatter(
        x = node_x,
        y = node_y,
        mode='text',
        text=rn_name,
        textfont = {'color':color, 'size':14},
        hoverinfo='skip',
        showlegend=False,
    )

    return node_trace, chord_text_trace, row_height
