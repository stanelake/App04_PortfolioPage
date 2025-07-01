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
            - Ph.D. in Mathematics and Computing, (Excellent) _Universitat de Barcelona_
            - MSc Data Science, (Distinction) _Edith Cowan University_ 
            - MSc Financial Engineering, (Higher Distinction) _WorldQuant University_
         """)   
st.subheader("Technical Skills:", anchor=False)    
import streamlit as st

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Stochastic Analysis")
    st.markdown("""
- Stochastic modelling  
- Stochastic differential equations  
- Monte Carlo simulation  
- Options pricing & Black-Scholes-Fischer model
- Lévy processes  
- Model calibration  
""")

with col2:
    st.markdown("#### Financial Engineering")
    st.markdown("""
- Financial data analysis  
- Asset pricing in Python  
- Risk management  
- Computational techniques  
- Econometrics  
- Programming in Python, R  
""")

with col3:
    st.markdown("#### Data Science")
    st.markdown("""
- Data manipulation & visualisation  
- Statistical software (R, Python)  
- Statistical inference  
- Research methodologies  
- Statistical analysis  
- Machine learning  
""")

# --- SOFT SKILLS ---- #
st.subheader("Soft Skills")
st.markdown("""
- Public speaking, 
- Collaboration, 
- Time management  
- Report writing in Microsoft Word, PowerPoint presentations  
- Project management  
""")

# --- PUBLICATIONS ---- #

st.subheader("Publications (Authors listed alphabetically):", anchor=False)
st.markdown("""
            - El-Khatib, Y., Makumbe, Z. S., & Vives, J. (2025). Decomposition of the option pricing formula for infinite activity jump-diffusion stochastic volatility models. *Mathematics and Computers in Simulation*, 231, 276-293.
            - El-Khatib, Y., Makumbe, Z. S., & Vives, J. (2024). Approximate option pricing under a two-factor Heston–Kou stochastic volatility model. *Computational Management Science*, 21(1), 3.
            - El-Khatib, Y., Goutte, S., Makumbe, Z. S., & Vives, J. (2023). A hybrid stochastic volatility model in a Lévy market. *International Review of Economics & Finance*, 85, 220-235.
            - El-Khatib, Y., Goutte, S., Makumbe, Z. S., & Vives, J. (2022). Approximate pricing formula to capture leverage effect and stochastic volatility of a financial asset. *Finance Research Letters*, 44, 102072.
            """)