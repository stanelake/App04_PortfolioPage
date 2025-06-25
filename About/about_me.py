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
    if st.button("📧 Contact Me"):
        show_contact_form()

# --- EXPERIENCE & QUALIFICATIONS
st.write("\n")
st.subheader("Qualifications:", anchor=False)
st.write("""
            - Ph.D. in Mathematics and Computing, _Universitat de Barcelona_
                - **Relevant Skills**: Stochastic modelling, Monte Carlo Simulation, data analysis, hypothesis testing, Python programming, public speaking, report writing, collaboration, time management.

         """)
         
st.write("""
            - MSc Data Science, _Edith Cowan University_ 
                - **Relevant Units**: Biostatistics, Time series forecasting, Applied Multivariate Statistics, Data Analysis and Visualisation
                - **Relevant Skills**: Data manipulation, data visualisation, statistical software (R, Python), statistical inference, research methodologies, statistical analysis, computational techniques, data modelling, machine learning, report writing in Microsoft Word, and PowerPoint presentation, project management

         """)         

st.write("""
            - MSc Financial Engineering, _WorldQuant University_
                - **Relevant Units**: Econometrics, Computational Finance, Machine Learning in Finance, Portfolio Theory and Asset Pricing, and Case studies in Risk Management
                - **Relevant Skills**: Financial data analysis, asset pricing in Python, machine learning, deep learning, Problem-Solving, Critical Thinking, Communication, Collaboration, Project Management, 

         """)

st.subheader("Hard Skills:", anchor=False)
st.write("""
            - Clear understanding of statistical principles and their applications
            - Strong Python programing skils
            - Experience training and deploying Machine Learning models
            """)