import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
import email
import os
import json
import platform
from hashlib import sha256
from cryptography.fernet import Fernet
import imaplib
import base64, hashlib

def generate_key():
    """
    Dynamically generate a consistent key based on system and user information.
    """
    # Gather system-specific information
    username = os.getlogin()  # Current logged-in username
    hostname = platform.node()  # Machine's hostname
    os_version = platform.system() + platform.release()  # OS name and version

    def gen_fernet_key(passcode:bytes) -> bytes:
        assert isinstance(passcode, bytes)
        hlib = hashlib.md5()
        hlib.update(passcode)
        return base64.urlsafe_b64encode(hlib.hexdigest().encode('latin-1'))
    
    passcode = f"{username}-{hostname}-{os_version}"
    key = gen_fernet_key(passcode.encode('utf-8'))
    return key

def encrypt_json_file(output_file, key, data):
    """
    Encrypt a JSON file using the generated key.
    """
    cipher = Fernet(key)
    
    # Encrypt the JSON content
    json_string = json.dumps(data)
    encrypted_data = cipher.encrypt(json_string.encode())
    
    # Write the encrypted data to the output file
    with open(output_file, 'wb') as file:
        file.write(encrypted_data)
    
    print(f"Encrypted data has been saved to {output_file}")

def decrypt_json_file(encrypted_file, key):
    """
    Decrypt an encrypted JSON file using the generated key.
    """
    cipher = Fernet(key)
    
    # Read the encrypted file
    with open(encrypted_file, 'rb') as file:
        encrypted_data = file.read()
    
    # Decrypt the data
    decrypted_data = cipher.decrypt(encrypted_data).decode()
    data = json.loads(decrypted_data)
    
    return data

def send_email(sender_email, sender_password, recipient_email, subject, body, attachment_path=None):
    try:
        # Set up the email components
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the email body
        msg.attach(MIMEText(body, 'plain'))

        # Attach a file if specified
        if attachment_path:
            if os.path.exists(attachment_path):
                filename = os.path.basename(attachment_path)
                with open(attachment_path, 'rb') as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={filename}'
                )
                msg.attach(part)
            else:
                print("Attachment file not found. Skipping attachment.")

        # Connect to the SMTP server
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        print("Email sent successfully!")

        # Close the connection
        server.quit()
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def get_email_credentials(account_name):
    base_dir='public/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    credentials = decrypt_json_file(f"{base_dir}email_credentials.enc", generate_key())
    return credentials[account_name]

def add_email_credentials(username,password):
    try:
        base_dir='public/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if os.path.exists(f"{base_dir}email_credentials.enc"):
            credentials = decrypt_json_file(f"{base_dir}email_credentials.enc", generate_key())
        else:
            credentials = {}
        credentials[username] = password
        encrypt_json_file(f"{base_dir}email_credentials.enc", generate_key(), credentials)
        return True
    except Exception as e:
        print(f"An error occurred in adding new credentials for email: {e}")
        return False

def all_emails():
    base_dir='public/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if os.path.exists(f"{base_dir}email_credentials.enc"):
        credentials = decrypt_json_file(f"{base_dir}email_credentials.enc", generate_key())
        return credentials
    else:
        return {}
    
def fetch_gmail_emails(username, password, max_emails=10):
    """
    Fetch emails from a Gmail account using IMAP.
    :param username: Your Gmail address.
    :param password: Your Gmail password (or app-specific password if using 2FA).
    :param max_emails: The maximum number of emails to fetch.
    :return: A list of email details (sender, subject, snippet of the body).
    """
    try:
        # Connect to Gmail's IMAP server
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(username, password)

        # Select the inbox
        imap.select("inbox")

        # Search for all emails
        status, messages = imap.search(None, "ALL")

        # Get the list of email IDs
        email_ids = messages[0].split()
        latest_email_ids = email_ids[-max_emails:]  # Get the most recent emails

        emails = []
        for email_id in reversed(latest_email_ids):
            # Fetch the email by ID
            res, msg = imap.fetch(email_id, "(RFC822)")
            for response in msg:
                if isinstance(response, tuple):
                    # Parse the email
                    msg = email.message_from_bytes(response[1])

                    # Decode the email subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")

                    # Decode the sender email
                    from_ = msg.get("From")

                    # Extract the email body (snippet)
                    body_snippet = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            # Get the plain text content
                            if part.get_content_type() == "text/plain":
                                body_snippet = part.get_payload(decode=True).decode()
                                break
                    else:
                        # For non-multipart emails
                        body_snippet = msg.get_payload(decode=True).decode()

                    # Add the email details to the list
                    emails.append({
                        "sender": from_,
                        "subject": subject,
                        "body": body_snippet[:100] + "..."  # Snippet of body
                    })

        # Logout from the IMAP server
        imap.logout()

        return emails

    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []