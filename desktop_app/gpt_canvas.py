
from gradio_client import Client
import ast
def generate_function(intent, query):
    example_function = '''def add_components_to_canvas(canvas_overlay, text=None):
#         """Example to dynamically add components to the canvas."""
        
#         # Include any and all imports needed for the components
#         from PyQt5.QtWidgets import QPushButton, QLabel, QWidget
#         from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QRect, QCoreApplication, pyqtProperty
#         from PyQt5.QtGui import QPainter, QColor

#         # Example button
#         button = QPushButton("Click Me", canvas_overlay)
#         button.move(100, 100)
#         button.clicked.connect(lambda: print("Button clicked!"))
#         canvas_overlay.add_component(button)

#         # Example label
#         label = QLabel("Dynamic Overlay Label", canvas_overlay)
#         label.move(100, 150)
#         label.setStyleSheet("color: white; font-size: 14px;")
#         canvas_overlay.add_component(label)''' 
   
    client = Client("yuntian-deng/ChatGPT")
    result = client.predict(
        inputs=f"Given the example function template: {example_function}, generate the functionContent, functionName and functionParameters with the exact same function definition to incorporate information from a LLM response {query} for user intent {intent}. Make the UI as clean, centered and simple as possible and always use WHITE/BRIGHT colors for texts. Omit any miscellaneous symbols and use English alphabets and numbers only.",
        top_p=1,
        temperature=1,
        chat_counter=0,
        chatbot=[],
        api_name="/predict"
    )
    result = result[0][0][1]
    result_code = result.split("python")[1].split("```")[0]
    # result_details = result.split("python")[1].split("```")[1]

    with open("canvas_update.py", "w") as f:
        f.write(result_code)
    
    functionName = result_code.split("def ")[1].split("(")[0]
    functionParameters = result_code.split("(")[1].split(")")[0].split(",")
    
    return functionName, result_code, functionParameters