# vijay-assignment
# Ticket Classification and Analysis Project

This project focuses on processing, analyzing, and modeling customer support ticket data. The primary goals are to clean and preprocess textual ticket data, engineer relevant features, and train machine learning models to predict the `issue_type` and `urgency_level` of tickets.

## Project Pipeline

The project follows these main steps:

1.  **Data Loading**: Loads ticket data from an Excel file (`ai_dev_assignment_tickets_complex_1000.xls`) into a pandas DataFrame.
2.  **Data Cleaning**:
    * Converts ticket text to lowercase.
    * Removes special characters from ticket text.
    * Handles missing values in `ticket_text` by imputing with empty strings.
    * Removes duplicate tickets based on `ticket_text`.
    * Standardizes `urgency_level` values (lowercase) and imputes missing values with 'unknown'.
3.  **Data Preparation**:
    * Performs text preprocessing on `ticket_text` including:
        * Tokenization
        * Stop word removal
        * Lemmatization
    * Encodes categorical target variables (`issue_type`, `urgency_level`) into numerical representations.
4.  **Feature Engineering**:
    * Creates Bag-of-Words (BoW) features.
    * Creates TF-IDF (Term Frequency-Inverse Document Frequency) features.
    * Calculates ticket length (number of words).
    * Performs sentiment analysis to derive a sentiment score.
    * Combines all engineered features into a final feature matrix.
5.  **Data Splitting**: Splits the dataset into training (80%) and testing (20%) sets, stratified by `issue_type`.
6.  **Model Training**:
    * Trains a `LogisticRegression` model to predict `issue_type`.
    * Trains a `SVC` (Support Vector Classifier) model to predict `urgency_level`.
7.  **Model Evaluation**: Evaluates the performance of both models on the test set using metrics like accuracy, precision, recall, and F1-score.
8.  **Data Exploration**:
    * Analyzes the distribution of products mentioned in the tickets.
    * Provides a sample list of potential complaint keywords (this part might require further development for automated keyword extraction).

## Files

* `vijay_assign_1.ipynb`: The main Jupyter Notebook containing all the code for the project.
* `ai_dev_assignment_tickets_complex_1000.xls`: The input Excel file containing the ticket data (expected to be in the same directory as the notebook or a correct path provided).

## Dependencies

The project relies on the following Python libraries:

* pandas
* numpy
* re (Regular Expressions)
* nltk (Natural Language Toolkit)
    * `punkt` tokenizer
    * `punkt_tab` tokenizer
    * `stopwords`
    * `WordNetLemmatizer`
    * `vader_lexicon` (for sentiment analysis)
* scikit-learn (sklearn)
    * `LabelEncoder`
    * `CountVectorizer`
    * `TfidfVectorizer`
    * `train_test_split`
    * `LogisticRegression`
    * `SVC`
    * `classification_report`, `accuracy_score`
* matplotlib (for plotting)

Ensure these libraries are installed in your Python environment. NLTK resources might need to be downloaded using `nltk.download('resource_name')`.

## How to Run

1.  Make sure you have Jupyter Notebook or JupyterLab installed.
2.  Ensure all dependencies listed above are installed.
3.  Place the `ai_dev_assignment_tickets_complex_1000.xls` file in the same directory as the `vijay_assign_1.ipynb` notebook, or update the file path in the "Data loading" section of the notebook.
4.  Open and run the `vijay_assign_1.ipynb` notebook cell by cell or by selecting "Run All".

## Results

* The `issue_type` classification model (Logistic Regression) achieved an accuracy of approximately 87.3% on the test set.
* The `urgency_level` classification model (SVC) achieved an accuracy of approximately 31.7% on the test set. Further improvements may be needed for this model.
