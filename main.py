import streamlit  as st

# --- PAGE SETUP ----#
about_page = st.Page(page = "About/about_me.py",
                     title = "About Me",
                     icon="👤",#"Images/home_icon.png",
                     default = True,)

project_1_page = st.Page(page="Projects/Cls_Proj_01.py",
                         title = "Iris Flower Classification",
                         icon = "🌼",)
project_2_page = st.Page(page="Projects/TS_Proj01_tickerPlot.py",
                         title = "Econometric Analysis",
                         icon = "📈",)

# ------ NAVIGATION SETUP WITH SECTIONS----- #
pg = st.navigation({
                    "Projects": [project_1_page,
                                 project_2_page],
                    "Info": [about_page]
                    }
                )

# ------- GLOBAL PAGE SETUP --------------- #
st.logo("Images/Logo4.png", size="large")
st.sidebar.text("Making numbers sing!")


# ------ RUN NAVIGATION ----#
pg.run()