import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier # import KNN
from sklearn.svm import SVC
import seaborn as sns

sns.set_theme()

def load_process_data(file_path, dropped_col=None):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    if dropped_col is not None:
        df = df.drop(columns=[dropped_col])
    array = df.values
    X = array[:, :-1]  # X is all columns except the last
    Y = array[:, -1]   # Y is the last column
    return X, Y, df

def data_split(X, Y, test_size =0.3):
    """
    Split the dataset into training and testing sets.
    Parameters:
    X : array-like, feature set 
    Y : array-like, target variable
    test_size : float, proportion of the dataset to include in the test split
    """
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    return X_train, X_test, Y_train, Y_test

def train_model( X_train, Y_train, model = ''):
    """
    Train the given model using the training data.
    Parameters:
    model : an instance of a machine learning model
    X_train : array-like, training feature set
    Y_train : array-like, training target variable
    """
    if model == 'decision_tree' or model == '':
        print("No model specified. Using Decision Tree Classifier by default.")
        model = DecisionTreeClassifier()
    elif model == 'logistic_regression':
        print("Training Logistic Regression model.")
        model = LogisticRegression(max_iter=200)
    elif model == 'random_forest':
        print("Training Random Forest Classifier model.")
        model = RandomForestClassifier()
    elif model == 'knn':
        print("Training K-Nearest Neighbors model.")
        model = KNeighborsClassifier()
    elif model == 'svm':
        print("Training Support Vector Machine model.")
        model = SVC(probability=True)
    model.fit(X_train, Y_train)
    return model
# create list of model options used in the function above
def st_print1():
    ## configure the app and add author and title
    st.set_page_config(page_title="Customer Purchase Prediction and Feature Impact Analysis", page_icon=":bar_chart:", layout="wide")
    st.title("Customer Purchase Prediction and Feature Impact Analysis")
    st.markdown(
        """
        <div class="project-by">Project by Zororo S. Makumbe</div>
        """,
        unsafe_allow_html=True
    )

st_print1()
X, Y, df = load_process_data('Projects/Cls_Proj2_Bike/data/BBC.csv')
def st_print2():
    st.write("""
            ## Introduction: 
            - This application addresses a common business decision problem: identifying which customers are most likely to purchase a product in order to support targeted marketing and resource allocation.
            - Using a real-world customer dataset, the app implements and compares several supervised machine learning models — Logistic Regression, Decision Trees, Random Forests, K-Nearest Neighbours, and Support Vector Machines — to predict customer bicycle purchases. 
                - Models are trained, validated, and evaluated using consistent performance metrics to enable fair comparison and model selection.
            - Beyond prediction accuracy, the application focuses on model interpretability and decision insight. 
                - A feature drop-out analysis is performed to assess the relative importance of input variables by measuring performance degradation when individual features are removed. 
                - This allows identification of the key drivers of customer purchasing behaviour, supporting explainable and actionable analytics. 
            - The app demonstrates an end-to-end applied machine learning workflow, from data preprocessing and model training to performance evaluation and interpretability, with direct relevance to business analytics and data-driven decision-making.
            """)
    st.write("""
            ### Data Overview:
            - Here is a preview of the dataset:""")
    st.dataframe(df.head())
    st.write("""
            - This data set contains {} instances and {} features (including the target variable).""".format(df.shape[0], df.shape[1]))
    st.write("""
            - Since we have more than 18000 instances, the models trained on this data will be robust and generalizable.""")
    st.write("- Features in the dataset are as follows:", df.columns[:-1].tolist())
    st.write("""
            ### Statistical Summary:""")
    st.write(df.describe())
    st.write("""### Class distribution:""")

    st.write(df.iloc[:, -1].value_counts())
    st.write("""
             - The data set consists of evenly balanced classes.
             - This ensures that the models trained on this data will not be biased towards a particular class.
             """)

    st.write("""
            ## Model Training and Evaluation:
            - The data is scaled using a standard scaler to ensure that all features are on the same scale.
            - The data is split into training and testing sets with a test size of 30%.
            - Several models are trained and evaluated on the test set and their accuracies are compared.
             
            ### Model Accuracies with All Features:""")
st_print2()

models = ['decision_tree', 'logistic_regression', 'random_forest', 'knn', 'svm']
models_ = ["Decision Tree", "Logistic Regression", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine"]
model_results = None
def st_print3():
    global model_results
    X_train, X_test, Y_train, Y_test = data_split(X, Y, test_size=0.3)
    # scale the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model_results = {}
    model = []
    for indx, model_name in enumerate(models):
        model.append(train_model(X_train_scaled, Y_train, model_name))
        model_results[models_[indx]] = model[-1].score(X_test_scaled, Y_test)    
    model_results = pd.DataFrame.from_dict(model_results, orient='index', columns=['Accuracy'])
    st.write(model_results)
    st.write("""            
            - From the results above, we can see that the Random Forest model performs the best with an accuracy of {:.2f}%. 
            - This model will be used for further analysis and feature importance evaluation.
            """.format(model_results.loc['Random Forest', 'Accuracy']*100))
    return model_results

model_results = st_print3()

dropped_model_results = None

def drop_out_analysis():  
    columns =   df.columns[:-1].tolist()
    st.write("## Drop-out Feature Analysis")
    base_performance = model_results.loc['Random Forest', 'Accuracy']
    dropped_model_results = {}
    for drop_col in columns:
        X_dropped, Y_dropped, _ = load_process_data('data/BBC.csv', dropped_col=drop_col)
        X_train_dropped, X_test_dropped, Y_train_dropped, Y_test_dropped = data_split(X_dropped, Y_dropped, test_size=0.3)
        scaler_dropped = StandardScaler()
        X_train_dropped_scaled = scaler_dropped.fit_transform(X_train_dropped)
        X_test_dropped_scaled = scaler_dropped.transform(X_test_dropped)
        dropped_model = train_model(X_train_dropped_scaled, Y_train_dropped, model= 'random_forest')
        dropped_model_results[drop_col] = (dropped_model.score(X_test_dropped_scaled, Y_test_dropped)- base_performance)/ base_performance * 100

    # plot as a barchart the accuracies
    dropped_model_results = pd.DataFrame.from_dict(dropped_model_results, orient='index', columns=['Accuracy Change (%)'])
    # sort the dataframe by accuracy change
    dropped_model_results = dropped_model_results.sort_values(by='Accuracy Change (%)')
    # plt using x=dropped_model_results.index, y=dropped_model_results['Accuracy Change (%)']
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x=dropped_model_results.index, 
           height=dropped_model_results['Accuracy Change (%)'])
    # Formatting
    ax.set_ylabel('Accuracy Change (%)')
    ax.set_title('Model accuracy change after dropping features')
    ax.set_xticklabels(dropped_model_results.index, rotation=45, ha='right')
    ax.grid(True)
    st.pyplot(fig)
    st.write("""
            - The bar chart above shows the change in model accuracy after dropping each feature.
             - Features that cause a significant drop in accuracy when removed are considered more important for the model's performance.
             - This analysis helps in understanding the relative importance of each feature in predicting customer purchases.
             - Removing less important features can also help in simplifying the model and reducing overfitting. Gender is the least important feature in this dataset. It may be worthwhile to remove it in future model training.
             - On the other hand, features like Income and Age have a significant impact on model performance, indicating their importance in predicting customer purchase behavior with Commute distance also being the most important.
             - However, due to the randomness in Random Forests, results may vary slightly with each run.

             ## Conclusion:
             - In this application, we successfully implemented and compared several machine learning models to predict customer bicycle purchases.
             - The Random Forest model emerged as the best performer, achieving the highest accuracy on the test set.
             - Through drop-out feature analysis, we identified the most influential features driving model performance, providing valuable insights into customer purchasing behavior.
             - This end-to-end workflow demonstrates the power of machine learning in addressing real-world business problems and highlights the importance of model interpretability for actionable decision-making.
             """)

    # Save dataframe to file
    dropped_model_results.to_csv('data/model_drop_out_analysis.csv')


drop_out_analysis()