import streamlit as st
from Forms.contact_me import show_contact_form

# --- HERO SECTION ---- #
col1, col2 = st.columns(2,
                        gap = "small",
                        vertical_alignment= "center")

with col1:
    st.image("./Images/Zororo_head.jpg", width=230)
with col2:
    st.title("Zororo Makumbe", anchor=False)
    st.write(
        """
            -	Seasoned mathematical data scientist with solid computing expertise 
            -	Proven ability to develop well-reasoned and integrated statistical solutions
            -	Keen on a data-driven career that employs computing and mathematical skills.
        """
        )
    if st.button(":mailbox_with_mail: Contact Me"):
        show_contact_form()

# --- EXPERIENCE & QUALIFICATIONS
st.write("\n")
st.subheader("Experience and Qualifications:", anchor=False)
st.write("""
            - Clear understanding of statistical principles and their applications
            - Strong Python programing skils
            - Experience training and deploying Machine Learning models
            - 
         """)

st.subheader("Hard Skills:", anchor=False)
st.write("""
            - Clear understanding of statistical principles and their applications
            - Strong Python programing skils
            - Experience training and deploying Machine Learning models
            - 
         """)

st.subheader("/soft Skills:", anchor=False)
st.write("""
            - Clear understanding of statistical principles and their applications
            - Strong Python programing skils
            - Experience training and deploying Machine Learning models
            - 
         """)