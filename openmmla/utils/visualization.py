import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pyecharts.commons.utils import JsCode


def format_time(hours, minutes, seconds):
    """Formats a time duration into a string in the format of hours, minutes, and seconds."""
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def weight_to_width(wt, min_wt=0.0, max_wt=0.5, min_width=1, max_width=20):
    """Converts a numerical weight value to a corresponding width value for visual representation in a graph."""
    wt = max(min_wt, min(max_wt, wt))
    width = min_width + (wt - min_wt) / (max_wt - min_wt) * (max_width - min_width)
    return width


def draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color="k",
                              font_family="sans-serif", font_weight="normal", alpha=None, bbox=None,
                              horizontalalignment="center", verticalalignment="center", ax=None, rotate=True,
                              clip_on=True, rad=0):
    """Custom function for drawing edge labels in NetworkX graphs.
    
    This function provides better handling of curved edges and label placement than the default
    NetworkX function, especially when multiple edges exist between nodes.
    """
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5 * pos_1 + 0.5 * pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad * rotation_matrix @ d_pos
        ctrl_mid_1 = 0.5 * pos_1 + 0.5 * ctrl_1
        ctrl_mid_2 = 0.5 * pos_2 + 0.5 * ctrl_1
        bezier_mid = 0.5 * ctrl_mid_1 + 0.5 * ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0

        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def get_pyecharts_js_functions():
    """Returns commonly used JavaScript functions for pyecharts visualizations."""
    return {
        'yaxis_min_js': JsCode("function(value){return Math.round(value.min * 0.8 * 1000) / 1000;}"),
        'yaxis_max_js': JsCode("function(value){return Math.round(value.max * 1.2 * 1000) / 1000;}"),
    }


def format_list_for_pyecharts(data):
    """Formats a list of numerical data for use with pyecharts.
    
    Converts None values to '0' and formats numbers to 4 decimal places.
    """
    return [f"{x:.4f}" if x is not None else '0' for x in data] 