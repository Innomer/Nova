from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout, QDialog, QMessageBox, QScrollArea, QWidget, QComboBox
from PyQt5.QtCore import Qt

def display_credentials(canvas_overlay, credentials, geometry):
    """
    Function to display existing user credentials on the canvas overlay.
    """
    # Add a Close button at the bottom
    close_button = QPushButton("Close", canvas_overlay)
    close_button.setGeometry(geometry[0]-60, 10, 60, 20)  # Adjust position and size as needed
    close_button.setStyleSheet("background-color: #555555; color: white; font-size: 14px; font-weight: bold;")
    close_button.clicked.connect(lambda: canvas_overlay.clear_components())

    # Add the Close button to the canvas
    canvas_overlay.add_component(close_button)
    
    if len(credentials)==0:
        label = QLabel("No credentials found.", canvas_overlay)
        label.move(50, 50)
        label.setStyleSheet("color: white; font-size: 12px;")
        canvas_overlay.add_component(label)
        return False
    y_offset = 50  # Starting vertical position for displaying credentials

    for username, password in credentials.items():
        label = QLabel(f"User: {username} | Password: {password}", canvas_overlay)
        label.move(50, y_offset)
        label.setStyleSheet("color: white; font-size: 12px;")
        canvas_overlay.add_component(label)
        y_offset += 30  # Move the next label down
    
    
    return True

def choose_email_id(canvas_overlay, credentials):
    """
    Function to display a dialog for choosing an email ID from the stored credentials.
    Returns the selected email ID, or None if canceled.
    """
    class ChooseEmailDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Choose Email ID")
            self.setFixedSize(300, 200)
            self.setStyleSheet("background-color: #2E2E2E; color: white;")

            # Layout
            layout = QVBoxLayout()

            # Instruction label
            label = QLabel("Select an email ID:", self)
            label.setStyleSheet("font-size: 14px;")
            layout.addWidget(label)

            # List of email IDs
            self.email_combo_box = QComboBox(self)
            self.email_combo_box.addItems(credentials.keys())
            layout.addWidget(self.email_combo_box)

            # OK and Cancel buttons
            button_layout = QHBoxLayout()
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)

            # Connect buttons
            ok_button.clicked.connect(self.accept)
            cancel_button.clicked.connect(self.reject)

            self.setLayout(layout)

        def get_selected_email(self):
            return self.email_combo_box.currentText()

    # Open the dialog
    dialog = ChooseEmailDialog()
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_selected_email()  # Return selected email ID
    return None  # Return None if canceled

def add_new_credential(canvas_overlay, credentials, update_callback):
    """
    Function to display a dialog for adding new user credentials.
    """
    
    class AddCredentialDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Add New Credential")
            self.setFixedSize(300, 200)
            self.setStyleSheet("background-color: #2E2E2E; color: white;")

            # Layout
            layout = QVBoxLayout()

            # Username input
            self.username_input = QLineEdit(self)
            self.username_input.setPlaceholderText("Enter username")
            layout.addWidget(QLabel("Username:", self))
            layout.addWidget(self.username_input)

            # Password input
            self.password_input = QLineEdit(self)
            self.password_input.setPlaceholderText("Enter password")
            self.password_input.setEchoMode(QLineEdit.Password)
            layout.addWidget(QLabel("Password:", self))
            layout.addWidget(self.password_input)

            # Add and Cancel buttons
            button_layout = QHBoxLayout()
            add_button = QPushButton("Add")
            cancel_button = QPushButton("Cancel")
            button_layout.addWidget(add_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)

            # Connect buttons
            add_button.clicked.connect(self.add_credential)
            cancel_button.clicked.connect(self.reject)

            self.setLayout(layout)
            self.success = False  # Flag to track if a credential was added

        def add_credential(self):
            username = self.username_input.text()
            password = self.password_input.text()

            if not username or not password:
                QMessageBox.warning(self, "Input Error", "Both username and password are required.")
                return

            if username in credentials:
                QMessageBox.warning(self, "Duplicate Entry", "Username already exists.")
                return

            credentials[username] = password  # Update the credentials dictionary
            update_callback(username, password)  # Refresh the UI or save changes
            self.success = True  # Mark success as True
            self.accept()  # Close the dialog

    # Open the dialog
    dialog = AddCredentialDialog()
    dialog.exec_()

    # Return the success flag
    return dialog.success
    
def display_emails(canvas_overlay, email_contents, geometry):
    """
    Function to display email contents dynamically on the canvas.
    :param canvas_overlay: The canvas where the components will be added.
    :param email_contents: A list of dictionaries representing email details.
                           Example: [{'sender': 'john@example.com', 'subject': 'Meeting Update', 'body': 'Hi, the meeting...'}]
    """

    # Create a scrollable area for the email list
    scroll_area = QScrollArea(canvas_overlay)
    scroll_area.setGeometry(20, 20, geometry[0]-40, geometry[1]-40)  # Adjust size and position as needed
    scroll_area.setStyleSheet("background-color: #2E2E2E; border: none; color: white;")
    scroll_area.setWidgetResizable(True)
    # Add a Close button at the bottom
    close_button = QPushButton("Close", canvas_overlay)
    close_button.setGeometry(geometry[0]-60, 10, 60, 20)  # Adjust position and size as needed
    close_button.setStyleSheet("background-color: #555555; color: white; font-size: 14px; font-weight: bold;")
    close_button.clicked.connect(lambda: canvas_overlay.clear_components())

    # Add the Close button to the canvas
    canvas_overlay.add_component(close_button)

    # Create a widget to hold the email content
    content_widget = QWidget()
    content_layout = QVBoxLayout(content_widget)
    content_widget.setStyleSheet("background-color: #2E2E2E; color: white;")

    # Dynamically add emails to the content layout
    for email in email_contents:
        sender = email.get('sender', 'Unknown Sender')
        subject = email.get('subject', 'No Subject')
        body_snippet = email.get('body', 'No Content')[:100] + '...'  # Show a snippet of the body

        # Create labels for sender, subject, and body snippet
        sender_label = QLabel(f"From: {sender}")
        sender_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        subject_label = QLabel(f"Subject: {subject}")
        subject_label.setStyleSheet("font-size: 12px; color: lightgray;")
        body_label = QLabel(f"Snippet: {body_snippet}")
        body_label.setStyleSheet("font-size: 12px; color: gray;")
        body_label.setWordWrap(True)

        # Add labels to the layout
        content_layout.addWidget(sender_label)
        content_layout.addWidget(subject_label)
        content_layout.addWidget(body_label)
        content_layout.addWidget(QLabel(" "))  # Spacer for readability

    # Set the content widget in the scroll area
    scroll_area.setWidget(content_widget)

    # Add the scroll area to the canvas
    canvas_overlay.add_component(scroll_area)
    