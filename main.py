import streamlit  as st

# --- PAGE SETUP ----#
about_page = st.Page(page = "About/about_me.py",
                     title = "About Me",
                     icon="👤",#"Images/home_icon.png",
                     default = True,)

project_1_page = st.Page(page="Projects/Proj_01.py",
                         title = "Dashboard",
                         icon = "",)

# ------ NAVIGATION SETUP WITH SECTIONS----- #
pg = st.navigation({
                    "Info": [about_page],
                    "Projects": [project_1_page]
                    }
                )

# ------- GLOBAL PAGE SETUP --------------- #
st.logo("Images/Logo3.png", size="large")
st.sidebar.text("Making numbers sing!")


# ------ RUN NAVIGATION ----#
pg.run()