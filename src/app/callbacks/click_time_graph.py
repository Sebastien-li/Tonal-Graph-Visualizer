""" Module for the callback when clicking on a node of the rhythm tree """
from dash import Output, Input, State, no_update, Patch
from time import time
from src.app.plotter import color_palette
from src.app.layout import visible_style
from src.rhythm_tree import RhythmTreeInteractive
from src.tonal_graph import TonalGraphInteractive

def click_time_graph_callback(app,harmonic_analyzer):
    """ Function for the callback when clicking on a node of the rhythm tree """

    @app.callback(
        Output('graph-time-graphs', 'figure',allow_duplicate=True),
        Output('graph-chord-graph', 'figure'),
        Output('div-chord-graph', 'style'),
        Input('graph-time-graphs', 'clickData'),
        State('trace-index', 'data'),
        State('graph-time-graphs', 'figure'),
        prevent_initial_call=True
    )
    def click_time_graph_callback(click_data, trace_index, figure):
        """ Callback for clicking on a node of the rhythm tree """

        time_graphs_patched_trace = Patch()
        if click_data is None:
            return no_update, no_update, no_update
        custom_data = click_data['points'][0]['customdata']
        custom_data = custom_data[0] if len(custom_data) == 1 else custom_data
        clicked_graph, clicked_index = custom_data[0], int(custom_data[1])
        if clicked_graph == 'rhythm_tree':
            dfs = list(harmonic_analyzer.rhythm_tree.depth_first_search())
            click_rhythm_tree(dfs, clicked_index)
            harmonic_analyzer.tonal_graph = TonalGraphInteractive(harmonic_analyzer.rhythm_tree)
            for i in trace_index['rhythm_fill']:
                rt_id = int(figure['data'][i]['customdata'][0][1])
                if rt_id == clicked_index:
                    time_graphs_patched_trace['data'][i]['fillcolor'] = color_palette['red']
                elif dfs[rt_id].selected:
                    time_graphs_patched_trace['data'][i]['fillcolor'] = color_palette['orange']
                else:
                    time_graphs_patched_trace['data'][i]['fillcolor'] = color_palette['light_blue']
            chord_fig = {}
            return time_graphs_patched_trace, chord_fig, visible_style
        return no_update, no_update, no_update

    def click_rhythm_tree(dfs, rt_index):
        """ Function to call when clicking on a node of the rhythm tree """
        node = dfs[rt_index]
        selected_parent = find_selected_parent(node)
        if selected_parent is None:
            children = node.depth_first_search()
            to_unselect = {n.id for n in children}
            to_select = {node.id}
        else:
            children = selected_parent.depth_first_search()
            to_select = {n.id for n in children if n.depth == node.depth}
            to_unselect = {selected_parent.id}
        update_selected_nodes(harmonic_analyzer.rhythm_tree, to_unselect, to_select)

    def find_selected_parent(node):
        while node is not None and not node.selected:
            node = node.parent
        return node

    def update_selected_nodes(node,to_unselect, to_select):
        if node.id in to_unselect:
            node.selected = False
        if node.id in to_select:
            node.selected = True
        for child in node.children:
            update_selected_nodes(child, to_unselect, to_select)
            child.parent = node
