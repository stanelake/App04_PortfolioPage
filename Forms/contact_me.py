import streamlit as st
import re
import requests

WEBHOOK_URL = ""

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
                st.error(":heavy_exclamation_mark: Email Servivce is not setup yet. Please try again later")
                st.stop()
                
            if not fname:
                st.error(":heavy_exclamation_mark: Please provide your first name")
                st.stop()
                
            if not sname:
                st.error(":heavy_exclamation_mark: Please provide your surname")
                st.stop()
                
            if not email:
                st.error(":heavy_exclamation_mark: Please provide your email. :email:")
                st.stop()
                
            if not message:
                st.error(":heavy_exclamation_mark: Please provide a message.")
                st.stop()
            data = {'email': email,
                    'first_name': fname,
                    'surname': sname,
                    'message': message}
            response = requests.post(WEBHOOK_URL, json=data)
            if response.status_code== 200:
                st.success(":white_check_mark: Message sent successfully")
            else:
                st.error(":heavy_exclamation_mark: ")