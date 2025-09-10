from bezierv.classes.bezierv import InteractiveBezierv
from bokeh.plotting import curdoc

# Define the initial control points for the single curve
initial_controls_x = [0.0, 0.25, 0.75, 1.0]
initial_controls_z = [0.0, 0.1, 0.9, 1.0]

# Create the manager instance with the initial curve
manager = InteractiveBezierv(
    controls_x=initial_controls_x,
    controls_z=initial_controls_z
)

# Add the plot layout to the document
curdoc().add_root(manager.layout)
curdoc().title = "Single Bézier Tool"