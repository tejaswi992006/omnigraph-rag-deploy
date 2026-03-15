"""Knowledge graph visualizer using Plotly."""
import plotly.graph_objects as go
import networkx as nx


def create_graph_viz(entities: list) -> go.Figure:
    """
    Build a Plotly figure showing *entities* as a small knowledge graph.

    Parameters
    ----------
    entities : list of str — entity names extracted from a retrieved chunk.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not entities:
        return _empty_figure("No graph data available")

    # Build a small star graph: central "Query Context" node + entity leaves
    G = nx.Graph()
    center = "Query Context"
    G.add_node(center)

    for entity in entities[:12]:           # cap at 12 nodes for readability
        G.add_node(str(entity))
        G.add_edge(center, str(entity))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
    )

    # Node traces — differentiate center vs leaves
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if node == center:
            node_color.append("#FF6B6B")
            node_size.append(22)
        else:
            node_color.append("#4ECDC4")
            node_size.append(14)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color="#fff"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="🕸️ Knowledge Graph — Extracted Entities",
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="#fafafa", size=11),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
        ),
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False,
                       font=dict(size=14, color="#aaa"))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=200,
    )
    return fig