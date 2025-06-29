import streamlit as st
import re
import requests

WEBHOOK_URL = "https://connect.pabbly.com/workflow/sendwebhookdata/IjU3NjYwNTZlMDYzMzA0Mzc1MjY1NTUzMDUxM2Ei_pc"

def is_valid_email(email):
    # Regex pattern for email validation
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.a-zA-Z0-9-.]+$"
    return re.match(email_pattern,email) is not None

@st.dialog("Contact Me")
def show_contact_form():
    with st.form("contact_form"):
        fname = st.text_input("First Name:")
        sname = st.text_input("Last Name:")
        email = st.text_input("Email Adress:")
        message = st.text_input("Your Message:")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if not WEBHOOK_URL:
                st.error("❗ Email Servivce is not setup yet. Please try again later")
                st.stop()
                
            if not fname:
                st.error("❗ Please provide your first name")
                st.stop()
                
            if not sname:
                st.error("❗ Please provide your surname")
                st.stop()
                
            if not email:
                st.error("❗ Please provide your email. 📧")
                st.stop()
                
            if not message:
                st.error("❗ Please provide a message.")
                st.stop()
            data = {'email': email,
                    'first_name': fname,
                    'surname': sname,
                    'message': message}
            response = requests.post(WEBHOOK_URL, 
                                     json=data)
            if response.status_code== 200:
                st.success("✅ Message sent successfully")
            else:
                st.error("❗ There was an error sending your message ❗")