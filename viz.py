import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

from kd import KDTree


def get_tree(x: int, y: int):
    np.random.seed(6)
    return KDTree(np.random.random((x, y)))


def plot_tree(tree: KDTree):
    DATA = np.asarray(tree.data)

    trace = go.Scatter(
        x=DATA[:, 0],
        y=DATA[:, 1],
        mode='markers',
        marker={
            'size': 10,
            'color': 'black'
        },
        text=tree.annotations,
        name='Data Points'
    )
    layout = {
        'shapes': [
            {
                'type': 'rect',
                'x0': x0,
                'y0': y0,
                'x1': x1,
                'y1': y1,
                'line': {
                    'color': 'rgb(50, 171, 96)' if x0 == x1 else 'rgb(250, 50, 50)',
                    'width': 2
                },
                'layer': 'below'
            }
            for (x0, y0), (x1, y1) in tree.bounding_boxes

        ],
        'showlegend': True
    }
    fig = {
        'data': [trace],
        'layout': layout,
    }
    iplot(fig)


if __name__ == '__main__':
    tree = get_tree(5, 2)
    plot_tree(tree)

    tree = tree.add(np.asarray([0.5, 0.5]))
    plot_tree(tree)

    tree = tree.delete(tree.data[0])
    plot_tree(tree)
