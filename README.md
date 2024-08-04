## End-to-End ML Project: DonorsChoose Application Screening


<!-- ------------------------------------------------------------------------------------------------ -->
### **Important Links:**
1. [Deployed Web Application](https://donors-choose-application-screening.onrender.com/) (Deployment is on a free cloud plan, which may result in slower initial load times)
2. [Tableau Dashboard](https://public.tableau.com/app/profile/iraban.dutta/viz/DonorsChooseProjectApproval/Dashboard1)
3. [Technical Blog](https://medium.com/@irabandutta.2020/donorschoose-project-approval-prediction-cb0a74bb932b)

<!-- ------------------------------------------------------------------------------------------------ -->
## Table of Contents:
1. [Introduction](#introduction)
2. [Basic Data Cleaning and EDA](#basic-data-cleaning-and-eda)
3. [Feature Engineering](#feature-engineering)
   - [Non-NLP Features (Project Metadata)](#non-nlp-features-project-metadata)
   - [Basic NLP Features](#basic-nlp-features)
   - [Advanced NLP Features](#advanced-nlp-features-word2vec)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Model Selection and Evaluation](#model-selection-and-evaluation)
6. [Deployment](#deployment)
7. [Run Web App Locally](#run-web-app-locally)


<!-- ------------------------------------------------------------------------------------------------ -->
## **Introduction:**

DonorsChoose is a U.S.-based nonprofit organization that helps public school teachers request funding for classroom projects. It connects donors with educators, enabling support for specific educational needs across the United States.

This project aims to predict whether a project proposal on DonorsChoose will be approved. The dataset comprises features related to project metadata and textual elements, including project titles, essays, and resource summaries. DonorsChoose launched the problem statement that we are solving here as a [Kaggle competition](https://www.kaggle.com/c/donorschoose-application-screening/overview).

### Project Highlights:
- Demonstrated comprehensive data science skills across the entire ML project lifecycle, including data cleaning, exploratory data analysis (EDA), feature engineering, hypothesis testing, model selection, evaluation and deployment.
- Employed advanced natural language processing (NLP) techniques to handle numerous textual features, enhancing model performance and interpretability.
- Developed the project using robust coding practices and object-oriented programming (OOP) principles, ensuring maintainability, scalability, and ease of understanding.
- Created an interactive Tableau dashboard for visualizing basic features, offering insightful data exploration and visualization.


<!-- ------------------------------------------------------------------------------------------------ -->
## **Basic Data Cleaning and EDA:**

The initial steps involved extensive data cleaning and exploratory data analysis (EDA). The following steps were performed:

- Removal of duplicate entries
- Handling missing values through smart imputation techniques
- Outlier detection using univariate analysis and necessary treatment
- Converted two categorical features with multiple possible categories per row from a comma-separated format to a multi-hot encoded format.

<!-- ------------------------------------------------------------------------------------------------ -->
## **Feature Engineering:**

### Non-NLP Features (Project Metadata):
- Temporal Features:
  - Year of submission
  - Month of submission
  - Day of the Week of submission
- Location Features:
  - U.S. state where the school is located
- Teacher Related Features:
  - Prefix of teachers (e.g., Mrs, Ms, Mr)
  - The number of submissions made by the teacher
  - Classification of the teacher as either a new or returning teacher on the DonorsChoose platform
- Project Related Features:
  - Grade level of the class for the project (e.g., preK-2, 3-5, 6-8, 9-12)
  - Project Subject Category
  - Project Subject SubCategory
- Resource Features (Aggregated at the project level):
  - Total number of unique resources requested in the project
  - Total count of all resources requested
  - Total count of expensive resources requested, highlighting the project's focus on high-cost items
  - Total price of all project resources combined
------------------------------------------------------------------------------------------------

### Basic NLP Features:
Before generating the Basic NLP features we performed the following steps:
- Text Preprocessing:
  - Basic Text Cleaning: Lowercasing, handling emojis, punctuations, escape sequences, stop-words, etc.
  - Tokenization: Splitting text into individual tokens (words or phrases) to facilitate analysis.
  - Stemming: Reduced words to their root forms to ensure consistency in feature extraction.
- Objectives of Text Preprocessing:
  - Vector Representation: Prepared the cleaned textual features for vectorization techniques such as Bag of Words (BoW), TF-IDF, and Word2Vec, enabling effective machine learning modeling.
  - Creation of Basic NLP Features: Extracted fundamental NLP features from the preprocessed text to capture essential characteristics and patterns.

A brief overview of the Basic NLP Features that were created:
- Count Features:
  - Count of Emojis, Emoticons, Punctuations
  - Count of Sentences, Words, Characters, Capitalized Words, Stop Words
- Ratio Features:
  - Ratio of Words to Sentences, Characters to Words, Capitalized Words to  Words, Stop Words to Words, etc.
- Normalized Rejection Score:
  - Presence of Words in Rejected Proposals: A normalized score (ranging from 0 to 1) indicating the presence of words that are exclusive to rejected proposals.
- Similarity Features (Between 2 Essays)
  - Features based on Similar Words: Count of common words
  - Features based on Token Length: Difference in the number of tokens, average length of tokens, longest substring ratio
  - Features based on FuzzyWuzzy Library: This library provides functions to calculate similarity scores between strings using Levenshtein distance
------------------------------------------------------------------------------------------------

### Advanced NLP Features (Word2Vec):
- Word2Vec Vector Representations
  - Gensim Model Creation and Training: Developed and trained Word2Vec models using the Gensim library to obtain vector representations for words in the textual features. This approach captures semantic relationships between words based on their context within the documents.
  - Vector Dimensions: Configured the Word2Vec model to create word vectors with 100 latent dimensions. This dimensionality strikes a balance between capturing meaningful word embeddings and managing computational efficiency.
- Document Representation
  - Average Word2Vec Calculation: Represented each document by calculating the average of all Word2Vec vectors for the words present in the document. This aggregated vector provides a summary representation of the document’s semantic content.
  - Statistical Features: Computed skewness and kurtosis for each document’s average Word2Vec vector to capture additional statistical properties.

<!-- ------------------------------------------------------------------------------------------------ -->
## **Hypothesis Testing:**

To validate the statistical significance of the features on the target variable, hypothesis testing was conducted as follows:

- Numerical Features: Used t-tests to compare the means of numerical features between approved and non-approved projects.
- Categorical Features: Employed Chi-squared tests to examine the associations between categorical features and the target variable.

This step ensured that the features we selected had a meaningful relationship with the target variable, enhancing the model's predictive power.

<!-- ------------------------------------------------------------------------------------------------ -->
## **Model Selection and Evaluation:**

The primary evaluation metric for this classification problem was the ROC-AUC score. Multiple models were trained across 17 iterations, experimenting with different combinations of the following factors:

- Model Type: Random Forest Classifier, XGBoost Classifier
- Features: Different sets of features including Non-NLP, Basic NLP, and Advanced NLP
- Sample Size: Varied proportions of the dataset on which the model was trained and evaluated (70-30 train-test split chosen)
- Balanced Target: Whether to balance the classes
- Hyperparameter Tuning: Optimization of model parameters

The final model was selected based on the highest ROC-AUC score, indicating the best performance in distinguishing between approved and non-approved projects.

### Summary:
<img width="995" alt="image" src="https://github.com/user-attachments/assets/a43e7d0a-2f73-4d9e-91dc-f8e1f6d3126e">

### ROC Curves:
![ROC_Curve](https://github.com/user-attachments/assets/10916d4e-8526-4980-b1b3-bbae015c3542)

<!-- ------------------------------------------------------------------------------------------------ -->
## **Deployment:**

The web application is deployed and accessible for public use. It includes a user-friendly interface where users can input project details and receive predictions on whether a project proposal is likely to be approved. Below are the details regarding the deployment:

- Framework: The web application is built using the Flask framework.
- Hosting: The application is hosted on a free cloud platform, providing a cost-effective solution for deployment.
- Model Integration: The trained machine learning model is integrated into the Flask server, allowing real-time predictions based on user input.
- User Interactivity: To simplify the user experience, we hardcoded some less important input features, such as the project submission year, month, etc.

<!-- ------------------------------------------------------------------------------------------------ -->
## **Run Web App Locally:**

1. Clone the git repository
```bash
git clone <repository-url>
```
2. Descend into the cloned directory and create a virtual environment
```bash
cd <repository-name>

python -m venv venv
```
3. Activate the virtual environment
- macOS:
```bash
source venv/bin/activate
```
- Windows:
```bash
venv\Scripts\activate
```
4. Install the required libraries
```bash
pip install -r requirements.txt
```
5. Start the Flask server
```bash
python app.py
```
6. Access the web application locally at http://127.0.0.1:5000/

<!-- ------------------------------------------------------------------------------------------------ -->



