import numpy as np
from animation_students import (
    render_graph,
    render_anim,
    make_surface,
    SPHERE,
    sphere,
)


X_surt, Y_surf, Z_surf = make_surface(
    min=SPHERE[0], max=SPHERE[1], function=sphere, step=0.1
)

render_graph(X_surt, Y_surf, Z_surf)

# replace empty lists with xy_data and z_data
x_data, z_data = [], []

render_anim(X_surt, Y_surf, Z_surf, x_data, z_data)
