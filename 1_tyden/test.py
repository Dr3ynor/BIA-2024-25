import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import random

# Create a Dash app
app = dash.Dash(__name__)

# Initialize data for the scatter plot
x_values = []
y_values = []
z_values = []

# Layout of the Dash app
app.layout = html.Div([
    dcc.Graph(id='live-3d-scatter', config={'scrollZoom': True}),  # Enable scroll zoom
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),  # Updates every second
])

# Callback to update the graph at every interval and keep the camera view
@app.callback(
    Output('live-3d-scatter', 'figure'),
    [Input('interval-component', 'n_intervals')],
    [State('live-3d-scatter', 'relayoutData')]  # Use relayoutData to store the camera view
)
def update_graph_live(n, relayout_data):
    # Simulate adding a new point in 3D space
    x_values.append(random.randint(0, 50))
    y_values.append(random.randint(0, 50))
    z_values.append(random.randint(0, 50))

    # Create the 3D scatter plot figure
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='markers',  # 'markers' means no lines, only points
            marker=dict(
                size=5,
                color=z_values,  # Color points based on the z-value
                colorscale='Viridis',  # Color scale for the points
                opacity=0.8
            )
        )],
        layout=go.Layout(
            title='Live 3D Scatter Plot',
            scene=dict(
                xaxis=dict(range=[0, 50]),
                yaxis=dict(range=[0, 50]),
                zaxis=dict(range=[0, 50]),
                aspectmode="cube"  # Make axes have equal scale
            )
        )
    )

    # Preserve the camera view if relayoutData exists
    if relayout_data and 'scene.camera' in relayout_data:
        fig.update_layout(scene_camera=relayout_data['scene.camera'])

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
