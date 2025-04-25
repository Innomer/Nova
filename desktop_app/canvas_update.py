
def add_components_to_canvas(canvas_overlay, text=None):
    """Dynamically add components to the canvas with a clean and simple design."""

    # Include any and all imports needed for the components
    from PyQt5.QtWidgets import QPushButton, QLabel, QWidget
    from PyQt5.QtCore import Qt

    # Example button
    button = QPushButton("Click Me", canvas_overlay)
    button.setStyleSheet("background-color: blue; color: white; font-size: 16px;")
    button.setGeometry(150, 120, 100, 40)  # Centered button
    button.clicked.connect(lambda: print("Button clicked!"))
    canvas_overlay.add_component(button)

    # Example label
    label = QLabel("Dynamic Overlay Label", canvas_overlay)
    label.setStyleSheet("color: white; font-size: 14px;")
    label.setGeometry(150, 70, 200, 30)  # Centered label
    canvas_overlay.add_component(label)
