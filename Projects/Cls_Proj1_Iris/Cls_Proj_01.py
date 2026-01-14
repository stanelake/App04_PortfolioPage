import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def user_input_features():
    '''
    Creates a Streamlit sidebar UI for user input and returns a DataFrame containing the selected feature values.

    The function provides four sliders in the sidebar to collect user input for sepal length, 
    sepal width, petal length, and petal width. These values are stored in a dictionary and 
    converted into a Pandas DataFrame for further processing.

    Returns:
        pandas.DataFrame: A DataFrame containing the selected feature values with a single row.

    Example:
        >>> df = user_input_features()
        >>> print(df)
           sepal_length  sepal_width  petal_length  petal_width
        0          5.4         3.4          1.3          0.2
    '''
    sepal_length = st.sidebar.slider('Sepal Length', 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.,4.4,3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.,6.9,1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1,2.5,.2)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    features  = pd.DataFrame(data, index=[0])
    return features

iris_data = datasets.load_iris()
X = iris_data.data
Y = iris_data.target

def dataPrep():
    """
    Prepares the Iris dataset by converting it into a Pandas DataFrame with labeled target values.

    This function takes the `iris_data` object (assumed to be loaded from `sklearn.datasets.load_iris()`) 
    and converts it into a DataFrame with feature columns, target labels, and mapped class names.

    Returns:
        pandas.DataFrame: A DataFrame containing the Iris dataset with the following columns:
            - Sepal length (cm)
            - Sepal width (cm)
            - Petal length (cm)
            - Petal width (cm)
            - target (numerical class: 0, 1, 2)
            - Target name (mapped class labels: 'setosa', 'versicolor', 'virginica')

    Example:
        >>> df = dataPrep()
        >>> print(df.head())
           sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target Target name
        0               5.1              3.5              1.4              0.2       0      setosa
        1               4.9              3.0              1.4              0.2       0      setosa
    """
    input = pd.DataFrame(iris_data.data, 
                         columns=iris_data.feature_names)
    input['target'] = iris_data.target
    input['Target name'] = input.target.map({0: 'setosa',
                                             1: 'versicolor',
                                             2: 'virginica'})
    return input

input = dataPrep()
def exploratoryDataAnalysis():
    """
    Performs exploratory data analysis (EDA) on the Iris dataset using Streamlit.

    This function:
    - Displays a preview of the first three rows of the dataset.
    - Provides insights about class separability based on petal characteristics.
    - Generates a pair plot using Seaborn to visualize relationships between features, 
      colored by target class.

    Assumptions:
        - `input` is a preprocessed Pandas DataFrame containing feature columns and a 
          categorical `Target name` column.
        - The function is intended for use in a Streamlit app.

    Returns:
        None: The function directly renders outputs in a Streamlit interface.

    Example:
        >>> exploratoryDataAnalysis()
        # Displays a DataFrame preview and a pair plot in the Streamlit app.
    """
    # Convert X into a pandas DataFrame
    st.subheader('Exploratory Data Analysis')
    st.write("""
             - The first stage of the application focuses on understanding the data before modelling, highlighting the role of exploratory analysis in responsible and effective machine learning.

            - Using the Iris dataset, the app presents an initial preview of the data and examines feature distributions and class separability, with particular attention to petal measurements. Through interactive visualisations, users can observe how different flower species separate in feature space, providing intuition about which variables are likely to be informative for classification.

            - A pairwise feature visualisation is generated to reveal relationships between input variables and their association with target classes. This step supports model interpretability and informed model selection, illustrating why certain models may perform better given the structure of the data.

            - This exploratory stage reinforces the importance of data understanding as a prerequisite for predictive modelling, aligning with best practices in applied machine learning and data-driven decision-making.
             
            #### DataFrame Preview
            """)
    st.write(input.head(3))
    st.write("""
            #### Pair Plots:
             - _Setosa_ seems to have high seperation in the petal characteristics.
             - _Versicolor_ and _Virginica_ have some overlap in those characteristics.
             - The plots suggest that we can come up with a trivial model 
            """)
    plot = sns.pairplot(input, hue='Target name')
    st.pyplot(plot)

def performanceScore(yActual, yPred):
    result = yActual==yPred
    return np.mean(result)

def compareModel(data = dataPrep()):
    """
    Trains and evaluates Random Forest and Logistic Regression models on the given dataset, 
    then compares their performance.

    This function:
    - Splits the dataset into training and testing sets (75%-25% split).
    - Trains a **Random Forest** model and a **Logistic Regression** model.
    - Computes performance scores on both training and testing sets.
    - Displays a comparison table of model performance in Streamlit.

    Args:
        data (pandas.DataFrame): The dataset, containing features and target labels.

    Returns:
        None: The function directly renders outputs in a Streamlit interface.

    Example:
        >>> df = dataPrep()
        >>> compareModel(df)
        # Displays model comparison results in the Streamlit app.
    """
    df_train, df_test = train_test_split(data, test_size=0.25)
    xTrain = df_train.drop(columns=['target', 'Target name']).values
    xTest = df_test.drop(columns=['target', 'Target name']).values
    yTrain = df_train.target.values
    yTest = df_test.target.values
    rf_model, _, _ = randForestModel(xTrain, yTrain)
    lr_mod, _, _ = logisticRegression(xTrain, yTrain)
    trainScore = [performanceScore(yTrain, rf_model.predict(xTrain)),
                    performanceScore(yTrain, lr_mod.predict(xTrain))]
    testScore = [performanceScore(yTest, rf_model.predict(xTest)),
                    performanceScore(yTest, lr_mod.predict(xTest))]
    perf_df = pd.DataFrame({'Train Score': trainScore,
                            'Test Score': testScore})
    perf_df = perf_df.rename(index={0: 'Rand. For.', 1: 'Log. Regr.'})
    st.write("""
        ## Camparing model performance: Model Performance Summary

        - This section compares the performance of the deployed models using both training and test accuracy scores, providing insight into model fit and generalisation. Displaying both metrics allows users to assess whether a model is learning meaningful patterns or overfitting to the training data.

        ### Prediction Consistency

        - For the user-defined input values, both models produce the same predicted flower species, indicating strong class separability for the selected features. While the predicted class is identical, the associated probability distributions may differ, reflecting differences in how each model represents uncertainty and decision boundaries.

        ### Key Takeaways

        - Comparable training and test scores suggest good generalisation performance.

        - Agreement between models increases confidence in the predicted outcome.

        - Differences in probability estimates highlight the trade-off between model simplicity (Logistic Regression) and model flexibility (Tree-based models).

        - Model comparison reinforces the importance of evaluating both performance metrics and interpretability, rather than relying on a single score. This comparative approach demonstrates how multiple models can be evaluated systematically to support robust, transparent, and defensible predictions.
             """)
    st.write(perf_df)

def randForestModel(x_array,y_array, user_df = None, classifier = None,
                    feature_names = None, target_names= None):
    if classifier is None:
        classifier = RandomForestClassifier(random_state=42)
    classifier.fit(x_array,y_array)
    if user_df is None:
        prediction, prediction_prob = None, None
    else:
        prediction = classifier.predict(user_df)
        prediction_prob = classifier.predict_proba(user_df)
    if not ((feature_names is None) and (target_names is None)):
        st.write("""
                 - This section introduces Random Forests, an ensemble learning approach in which multiple decision trees work together to produce more accurate and robust predictions than a single tree alone.

                - Rather than relying on one “optimal” tree, the model aggregates the decisions of many trees trained on different subsets of the data. This reduces overfitting and improves generalisation, particularly in datasets where individual trees may capture noise.

                - The application allows users to visualise individual trees within the forest using the slider below, making the ensemble structure transparent. By exploring how different trees make slightly different decisions, users gain insight into why ensemble methods tend to outperform single models and how collective decision-making improves predictive stability.

                - This interactive visualisation supports explainable AI, helping users understand not only what the model predicts, but how those predictions are formed.
                 """)
        tree_index = st.slider("Select Tree Index", 0, len(classifier.estimators_) - 1, 0)

        # Improved visualization settings
        fig, ax = plt.subplots(figsize=(14, 8), dpi=120)  # Bigger and sharper
        tree.plot_tree(classifier.estimators_[tree_index], 
                  feature_names=feature_names, 
                  class_names=target_names, filled=True, ax=ax)

        # Display in Streamlit
        st.pyplot(fig)

    return classifier, prediction, prediction_prob

def logisticRegression(x_array,y_array, user_df=None, lr_mod = None):
    """
    Trains a Logistic Regression model and optionally makes predictions.

    If `lr_mod` is not provided, a new Logistic Regression model is initialized and trained 
    using the given feature and target arrays. If `user_df` is provided, the function 
    returns predictions and their associated probabilities.

    Args:
        x_array (numpy.ndarray or pandas.DataFrame): Feature matrix for training.
        y_array (numpy.ndarray or pandas.Series): Target labels for training.
        user_df (pandas.DataFrame, optional): DataFrame containing new observations 
                                              for prediction. Defaults to None.
        lr_mod (LogisticRegression, optional): Pre-trained Logistic Regression model.
                                               If None, a new model is trained.

    Returns:
        tuple: (trained Logistic Regression model, predictions, prediction probabilities)
            - `LogisticRegression` : The trained model.
            - `numpy.ndarray or None` : Predictions if `user_df` is provided, else `None`.
            - `numpy.ndarray or None` : Prediction probabilities if `user_df` is provided, else `None`.

    Example:
        >>> lr_model, preds, probs = logisticRegression(X_train, y_train, X_test)
        >>> print(preds)  # Output predicted labels
        >>> print(probs)  # Output class probabilities
    """
    if lr_mod is None:
        lr_mod = LogisticRegression(max_iter=100,
                                    C=1.0)
    lr_mod.fit(x_array, y_array)
    if user_df is None:
        prediction, prediction_prob = None, None
    else:
        prediction = lr_mod.predict(user_df.values)
        prediction_prob = lr_mod.predict_proba(user_df.values)
    return lr_mod, prediction, prediction_prob
# Custom CSS for the header
st.markdown(
    """
    <style>
    .project-title {
        font-size: 36px !important;
        font-weight: bold !important;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .project-by {
        font-size: 20px !important;
        font-style: italic;
        text-align: center;
        color: #5D6D7E;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Header with the project title and your name
st.markdown(
    """
    <div class="project-title">Interactive Model Selection and Prediction Confidence Analysis: Iris Flower Classifier</div>
    """,
    unsafe_allow_html=True
)
st.write("""
         ## Introduction: 
    - This application explores a fundamental applied machine learning question: how model choice affects predictions and confidence, even when using the same input data.

    - Using a well-known multiclass classification problem, the app trains and deploys a Decision Tree and Logistic Regression model to predict flower species based on four input features. Users can dynamically select which model to deploy and input custom feature values, enabling hands-on exploration of model behaviour.

    - A key focus of the application is comparative model analysis. By evaluating prediction outcomes and associated probability estimates, the app highlights differences in decision boundaries, model confidence, and interpretability between linear and tree-based models.

    - By allowing users to experiment with artificial and user-defined inputs, the application demonstrates how model assumptions influence predictions, supporting informed model selection and transparent AI deployment. The app provides an intuitive interface for understanding prediction uncertainty, making it particularly relevant for education, explainable AI, and applied analytics.
         """)

st.sidebar.header('User Input Parameters')
with st.sidebar:
    st.sidebar.subheader('Choose the process you want:')
    eda_check = st.checkbox("Exploratory data Analysis?")
    rf_check = st.checkbox("Random Forest Prediction?")
    lr_check = st.checkbox("Logistic Regression?")
    modComp_check = st.checkbox("Compare Random Forest and Logistic Regression models?")
    st.sidebar.subheader('Choose your inputs:')
    df = user_input_features()



st.subheader('Class labels and their index numbers')
st.write(list(iris_data.target_names))
st.write("""
- The Iris data set has a total of 3 classes.
- The class labels are 0, 1, and 2 which correspond to 'setosa', 'versicolor', and 'virginica' respectively.
- The model will predict the class label based on the input features.
         """)


if eda_check:
    exploratoryDataAnalysis()
if rf_check or lr_check:
    rf_results = [None]*4
    lr_results = [None]*4
    colz = ['Prediction'] + [name.capitalize() + ' Prob.' for name in iris_data.target_names]
    if rf_check: 
        rf_results = [] 
        st.subheader('Random Forest Model')
        rf_model, pred, pred_prob =randForestModel(X,Y,df,
                                                   feature_names=iris_data.feature_names,
                                                   target_names=iris_data.target_names)
        rf_results.append(iris_data.target_names[pred])
        rf_results = rf_results + list(pred_prob[0])

    if lr_check:
        lr_results = []
        st.subheader('Logistic Regression Model: Interpretable Baseline and Probability-Driven Decisions')
        st.write("""
        - This section uses Logistic Regression as a transparent, interpretable baseline model for classification. The model learns a linear relationship between input features and class outcomes, making it particularly useful for understanding how individual variables influence predictions.

        - Once trained, the model can be applied to user-defined inputs, producing both class predictions and associated probabilities. These probability estimates provide insight into the model’s confidence, supporting more informed decision-making rather than relying solely on hard classifications.

        - By comparing Logistic Regression results with tree-based models elsewhere in the app, users can observe how model assumptions affect predictions, confidence, and interpretability. This highlights the trade-off between model simplicity and flexibility, an important consideration in applied machine learning and responsible AI deployment.
                 """)
    st.write("""
                #### Prediction Results and Model Confidence

            - Each selected model generates a predicted flower species based on the input feature values. Alongside the predicted class, the application displays the probability associated with each possible class, providing insight into the model's confidence.

            - Presenting prediction probabilities allows users to move beyond a single label and assess prediction certainty and ambiguity, particularly in cases where classes overlap. This supports more informed interpretation of results and highlights how different models may express confidence differently, even when making the same prediction.

            - By combining predicted outcomes with probability estimates, the app reinforces best practices in transparent and explainable AI, helping users understand not only what the model predicts, but how strongly it supports that prediction.
                """)
    results = pd.DataFrame(columns=colz,
                           index = ['Random Forest', 'Logistic Regr'])
    if not (rf_results[0] is None):
        for val, col in zip(rf_results, colz):
            results.at['Random Forest', col] = val
    if not (lr_results[0] is None):            
        for val, col in zip(lr_results,colz):
            results.at['Logistic Regr', col] = val

    st.write(results)

if  modComp_check:        
    compareModel()

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Credits: <a style='display: block; text-align: center;' href="https://youtu.be/8M20LyCZDOY?list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE" target="_blank">The Data Professor</a></p>
<p><a style='display: block; text-align: center;' href="https://discuss.streamlit.io/t/streamlit-footer/12181/3" target="_blank">Streamlit Discussion Board</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)