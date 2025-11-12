# Car Evaluation Decision Tree Classifier (Streamlit App)

## Project Overview

This project focuses on predicting the **acceptability of cars** based on multiple features such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

A **Decision Tree Classifier** is implemented to analyze the relationships between these attributes and classify cars into four categories — *Unacceptable, Acceptable, Good,* and *Very Good.*

To enhance reliability, **K-Fold Cross-Validation** is used for model evaluation. A clean and interactive **Streamlit web application** is built, allowing users to enter car attributes and instantly view the predicted category.

---

## Objectives

* Build a Decision Tree model to classify car acceptability.
* Encode categorical variables for model compatibility.
* Apply **K-Fold Cross-Validation** for accurate model assessment.
* Integrate a **Streamlit-based UI** for easy, real-time predictions.

---

## Dataset Information

* **Source:** UCI Machine Learning Repository – *Car Evaluation Dataset*
* **Attributes:**

  * Buying Price
  * Maintenance Cost
  * Number of Doors
  * Passenger Capacity
  * Luggage Boot Size
  * Safety Level
* **Target Variable:** Car Acceptability (`Unacc`, `Acc`, `Good`, `VGood`)

---

## Technologies Used

* **Programming Language:** Python
* **Libraries & Tools:**

  * pandas, numpy
  * scikit-learn
  * ucimlrepo
  * streamlit
  * matplotlib (optional for visualization)

---

## Project Workflow

### 1. Data Loading

Dataset fetched using `fetch_ucirepo` from the `ucimlrepo` library.

### 2. Data Preprocessing

* Encoding categorical features such as buying, maint, doors, persons, lug_boot, and safety.
* Mapping all text-based columns into numeric form to ensure compatibility with the model.

### 3. Model Training

* The model is trained using **DecisionTreeClassifier(criterion='entropy')**.
* Cross-validation (5-fold) is applied to assess the model’s performance stability.

### 4. Model Evaluation

* Accuracy score and average cross-validation accuracy are computed.
* Model interpretability is achieved through visualization using `plot_tree`.

### 5. Streamlit App Development

* Streamlit is used to build a user interface where users can enter car details.
* On submission, the trained model predicts the car acceptability category instantly.

---

## How to Run Locally

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Car-Evaluation-Decision-Tree.git
cd Car-Evaluation-Decision-Tree
```

### Step 2: Install Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or manually install the dependencies:

```bash
pip install pandas numpy scikit-learn ucimlrepo streamlit matplotlib
```

### Step 3: Run the Jupyter Notebook

For training and testing the model:

```bash
jupyter notebook decisiontreeclassi.ipynb
```

### Step 4: Launch the Streamlit App

```bash
streamlit run car_evaluation_app.py
```

### Step 5: Open in Browser

After running the above command, Streamlit will display a local URL, typically:

```
http://localhost:8501
```

Open this link in your browser to access the web app.

---

## Results and Insights

* The Decision Tree model achieved strong accuracy for predicting car evaluation categories.
* Cross-validation confirmed the model’s robustness and reduced overfitting.
* The Streamlit app provided an interactive and user-friendly experience for real-time predictions.

---

## Streamlit App Preview

*(Add a screenshot of your Streamlit app below — upload the image in your repo and replace the path)*

![Streamlit App Screenshot](images/streamlit_app.png)


## Folder Structure

```
Car-Evaluation-Decision-Tree/
│
├── decisiontreeclassi.ipynb        # Jupyter Notebook with model code
├── car_evaluation_app.py           # Streamlit app script
├── requirements.txt                # Dependencies file
├── README.md                       # Project documentation
└── /images/streamlit_app.png       # App screenshot (optional)
```

---

## Author

**Ayushi Parkhe**
Data Science | Machine Learning | AI Enthusiast

* **LinkedIn:** [linkedin.com/in/your-link](https://www.linkedin.com/in/ayushi-parkhe-bb2404240/)
* **GitHub:** [github.com/yourusername](https://github.com/AYUSHIPARKHE)


## Acknowledgment

Dataset source: [UCI Machine Learning Repository – Car Evaluation Dataset](https://archive.ics.uci.edu/ml/datasets/car+evaluation)

