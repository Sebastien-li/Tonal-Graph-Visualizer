""" Plotter module for the app. """
import base64
from fractions import Fraction
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rhythm_tree import RhythmTreeInteractive
from src.music_theory_classes import Pitch, Key
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

rhythm_unicode = {6: '\U0001D15D.',
                  4: '\U0001D15D', 3 : '\U0001D15E.',
                  2: '\U0001D15E', 1: '\U0001D15F',
                  0.5: '\U0001D160', 1.5:'\U0001D15F.',
                  0.25: '\U0001D161', 0.75:'\U0001D160.',}

def data_to_array(type_str, index):
    """ Convert the custom data to an array """
    if np.size(index) == 1:
        return np.array([(type_str, index)])
    else:
        return np.array([(type_str, x) for x in index])

def plot_score(harmonic_analyzer, plot_graph = True):
    """ Plot the score of the piece. """
    svg_data = harmonic_analyzer.svg_data
    note_graph = harmonic_analyzer.note_graph

    svg_doc = harmonic_analyzer.svg_doc

    encoded_svg_data = base64.b64encode(svg_data.encode()).decode()

    fig = go.Figure()
    graph_scale = .7
    graph_width = svg_doc.svg_width * graph_scale
    graph_height = svg_doc.svg_height * graph_scale
    graph_size = (graph_width, graph_height)

    figure_width = 1900
    figure_height = graph_height + 100

    fig.add_layout_image(x=0, sizex=graph_width, y=0, sizey=graph_height, xref="x", yref="y",
                         layer="below",
                         source='data:image/svg+xml;base64,' + encoded_svg_data)
    fig.update_xaxes(showgrid=False, visible=False, range=[0,figure_width])
    fig.update_yaxes(showgrid=False, visible=False, range=[figure_height-100,-100])
    fig.update_layout(height=figure_height, width=figure_width, margin={'l':0, 'r':0, 'b':0, 't':0},
                      plot_bgcolor='white', dragmode='pan',
                      legend = {'yanchor':'top', 'y':0.99, 'xanchor':'left', 'x':0.01})

    edge_show_legend = {'onset':True, 'during':True, 'follow':True, 'silence':True}

    for i, edge_index in enumerate(note_graph.edge_index):
        edge_attr = note_graph.edge_attr[i]['type']
        src = note_graph.nodes[edge_index[0]]
        dst = note_graph.nodes[edge_index[1]]
        src_type = 'end' if edge_attr != "onset" else 'middle'
        dst_type = 'start' if edge_attr != "onset" else 'middle'
        src_x, src_y = svg_doc.get_note_coords(src['note_id'], graph_size, src_type)
        dst_x, dst_y = svg_doc.get_note_coords(dst['note_id'], graph_size, dst_type)
        trace = go.Scatter(
            x = [src_x, dst_x, None],
            y = [src_y, dst_y, None],
            line = {"color": color_dict[edge_attr]},
            hoverinfo='none',
            mode='lines',
            opacity=0.5 if plot_graph else 0,
            showlegend=edge_show_legend[edge_attr],
            legendgroup='Show Note graph',
            legendgrouptitle={'text':'Click here to show/hide the note graph'},
            name=edge_attr
        )
        fig.add_trace(trace)
        edge_show_legend[edge_attr] = False

    return fig

def plot_time_graph(harmonic_analyzer, xmin=None, xmax=None):
    """ Create a plotly figure with the temporal graph of the analysis. """
    note_graph = harmonic_analyzer.note_graph
    rhythm_tree = harmonic_analyzer.rhythm_tree
    tonal_graph = harmonic_analyzer.tonal_graph

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[2,1,1],
                        subplot_titles=('Note graph', 'Rhythm tree', 'Key selection'))
    fig.update_layout(
        showlegend=True, dragmode='pan',
        height = 1000, width = 1900,
        title_text = "Temporal graphs",
        legend_tracegroupgap=110,
    )

    # X axis layout
    score = note_graph.score
    if xmin is None:
        xmin = float(score[0].measures[0].start.t) - 1
    if xmax is None:
        last_measure = score[0].measures[min(11, len(score[0].measures)-1)]
        xmax = float(last_measure.end.t) + 1

    fig.update_xaxes(range = [xmin, xmax],
                     tickvals = [x.start.t for x in score[0].measures],
                     ticktext = [x.number for x in score[0].measures],
                     matches = 'x')

    fig.update_layout(xaxis_showticklabels=True,
                      xaxis2_showticklabels=True,
                      xaxis3_showticklabels=True,xaxis3_title = 'measure number',
                      yaxis3_showticklabels=False, coloraxis_showscale=False)

    #Note graph
    edge_trace, node_traces = make_note_graph_trace(note_graph)
    trace_index = {'note_edge':[], 'note_node':[],
                   'rhythm_fill':[], 'rhythm_border':[], 'rhythm_text':[],
                   'key_fill':[], 'key_border':[], 'key_text':[]}
    trace_i = 0
    for trace in edge_trace:
        fig.add_trace(trace, row=1, col=1)
        trace_index['note_edge'].append(trace_i)
        trace_i += 1

    fig.add_trace(node_traces, row=1, col=1)
    trace_index['note_node'].append(trace_i)
    trace_i+=1

    # Rhythm tree
    rt_fill, rt_selected_fill, rt_border, rt_text = make_rhythm_tree_trace(rhythm_tree)
    for trace in rt_fill:
        fig.add_trace(trace, row=2, col=1)
        trace_index['rhythm_fill'].append(trace_i)
        trace_i += 1
    for trace in rt_selected_fill:
        fig.add_trace(trace, row=2, col=1)
        trace_index['rhythm_fill'].append(trace_i)
        trace_i += 1
    for trace in rt_border:
        fig.add_trace(trace, row=2, col=1)
        trace_index['rhythm_border'].append(trace_i)
        trace_i += 1
    fig.add_trace(rt_text, row=2, col=1)
    trace_index['rhythm_text'].append(trace_i)
    trace_i += 1

    # Tonal graph
    # node_trace, chord_text_trace, row_height = make_tonal_graph_trace(tonal_graph)
    # fig.add_trace(node_trace, row=3, col=1)
    # trace_index['tonal_node'].append(trace_i)
    # trace_i += 1
    # fig.add_trace(chord_text_trace, row=3, col=1)
    # trace_index['tonal_text'].append(trace_i)
    # trace_i += 1

    #Key selector
    key_selector_data = make_key_selector_trace(tonal_graph)
    fig.add_trace(key_selector_data, row=3, col=1)


    # Y axis layout
    fig.update_yaxes(showgrid = False, row=1, col=1, title='pitch space')
    fig.update_yaxes(showgrid = True,
                     zeroline = False,
                     tickvals = [-0.75,0.25,1.25,2.25],
                     ticktext = ['1/2','1','2','4'],
                     row=2, col=1,
                     title = 'subdivision')
    # fig.update_yaxes(zeroline = False,
    #             tickvals = row_height-0.5,
    #             ticktext = ['C','C#', 'D','E-', 'E', 'F', 'F#', 'G', 'A-', 'A','B-', 'B', ''],
    #             row=3, col=1)

    fig.update_yaxes(zeroline = False,
                     range = [-3,3],
                     row=3, col=1)


    return fig, trace_index

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
        customdata = data_to_array('note_graph', nodes['id']),
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
    for i, node in enumerate(rhythm_tree.depth_first_search()):
        if node.depth == 0:
            continue
        x0 = node.onset
        x1 = node.offset - 0.1
        y0 = np.log2(float(node.subdivision)) if node.depth != 0 else 3
        y1 = np.log2(float(node.subdivision))+0.5

        # Rectangle fill
        if node.selected:
            rectangle_fill_selected_traces.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                hoverinfo='none' ,
                fillcolor=color_palette['orange'],
                line={'color':color_palette['gray']},
                customdata=data_to_array('rhythm_tree',i),
                showlegend=False,
                name = "Rhythm tree",
                legendgroup='rhythm_tree',
                ))
        else:
            rectangle_fill_traces.append(go.Scatter(
                x = [x0,x1,x1,x0,x0,None],
                y = [y0,y0,y1,y1,y0,None],
                fill='toself',
                mode='lines',
                hoverinfo='none',
                fillcolor=color_palette['light_blue'],
                line={'color':color_palette['gray']},
                customdata=data_to_array('rhythm_tree',i),
                showlegend=False,
                name = "Rhythm tree",
                legendgroup='rhythm_tree',
                ))


        # Text
        text_x.append((x0+x1)/2)
        text_y.append((y0+y1)/2)
        dur = node.duration / node.duration_divisor
        if dur in rhythm_unicode :
            text.append(f"{rhythm_unicode[dur]}")
        else:
            text.append(f"{dur}")

    text_trace = go.Scatter(
        x=text_x,
        y=text_y,
        text=text,
        mode='text',
        textposition='middle center',
        hoverinfo='skip',
        textfont={
            'size':32,
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

def make_key_selector_trace(tonal_graph):
    path_info = tonal_graph.shortest_path
    xticks = path_info['onset'] + path_info['duration']/2
    yticks = np.arange(-42.5,42.5,0.5)
    ytick_labels = []
    for y in yticks:
        if y.is_integer():
            diatonic = (y*4)%7
            chromatic = (y*7)%12
            mode = "M"
        else:
            diatonic = ((y+0.5)*4-2)%7
            chromatic = ((y+0.5)*7-3)%12
            mode = "m"
        tonic = Pitch(int(diatonic),int(chromatic))
        ytick_labels.append(tonic.name if mode == "M" else tonic.name.lower()+"m")
    ytick_labels = np.array(ytick_labels)

    heatmap_data = np.zeros((len(yticks),len(path_info)))
    for i, node in enumerate(path_info):
        diatonic = node['tonic_diatonic']
        chromatic = node['tonic_chromatic']
        mode = node['mode']
        tonic = Pitch(diatonic,chromatic)
        name = tonic.name if mode == "M" else tonic.name.lower()+"m"
        y_index = np.where(ytick_labels == name)[0][0]
        heatmap_data[y_index,i] = 1

    text = np.stack([ytick_labels]*len(path_info),axis=1)
    data = go.Heatmap(z=heatmap_data, text=text, x=xticks, y=yticks,
                      hoverinfo='text',
                      colorscale=[(0, color_palette['light_blue']), (1, color_palette['orange'])],
                      texttemplate='%{text}', xgap=1, ygap=1, showscale=False,)
    return data