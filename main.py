import streamlit  as st

# --- PAGE SETUP ----#
about_page = st.Page(page = "About/about_me.py",
                     title = "About Me",
                     icon="👤",#"Images/home_icon.png",
                     default = True,)
project_1_page = st.Page(page="Projects/Cls_Proj1_Iris/Cls_Proj_01.py",
                         title = "Iris Flower Classification",
                         icon = "🌼",)
project_2_page = st.Page(page="Projects/Cls_Proj2_Bike/Cls_Proj_02.py", # This is a bike purchase project
                         title = "Bike Purchase Prediction",
                         icon = "🚲",)
project_3_page = st.Page(page="Projects/TS_Proj1_Analysis/TS_Proj01_tickerPlot.py",
                         title = "Econometric Analysis",
                         icon = "📈",)

# ------ NAVIGATION SETUP WITH SECTIONS----- #
pg = st.navigation({
                     "Info": [about_page],
                    "Projects": [project_1_page,
                                 project_2_page,
                                 project_3_page]
                    }
                )

# ------- GLOBAL PAGE SETUP --------------- #
st.logo("Images/Logo4.png", size="large")
# st.sidebar.text("Making numbers sing!")


# ------ RUN NAVIGATION ----#
pg.run()