## Group Members:
1) Ragasree Katam  - A20552861
2) Sesha Sai Sushmith Muchintala  -A20536372
3) Udhay Chander Bharatha  -A20518701
## LASSO Regression with Homotopy Method

This project implements the LASSO regularized regression model using the Homotopy method from first principles in Python. The implementation is done using NumPy/Scipy and does not rely on built-in models from libraries such as SciKit Learn. Instead, SciKit Learn is used only for generating test data and comparisons.

## Overview

The LASSO (Least Absolute Shrinkage and Selection Operator) is a regression method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces. The Homotopy method gradually decreases the regularization parameter (lambda) from a high value, tracking the solution path of the coefficients.



## Setup

1. **create virtual env and activate it :**

    python3 -m venv venv  #Linux/Mac
    python -m venv venv   # Windows
    source venv/bin/activate  # Linux/Mac  
    venv\Scripts\activate     # Windows   

2. **Install the required dependencies, including NumPy, Pandas, Matplotlib, and Seaborn**
    python.exe -m pip install --upgrade pip
    pip install -r requirements.txt

3. **To test the implementation, run the tests using:**

    pytest LassoHomotopy/tests/test_LassoHomotopy.py

4. **For Visualisation:**
    Run LassoHomotopy/train.ipynb file.

4. **Exit:**
    deactivate


**Questions & Answers:**

* What does the model you have implemented do and when should it be used? *

=> The model I built is designed to simplify complex datasets by identifying the most important predictors. Imagine you’re working with a dataset where you have dozens of features—like customer demographics, purchase history, and website behavior—but only a handful truly impact sales. This model uses a technique called LASSO regression with a "path-following" approach (Homotopy method) to automatically zero in on those key features while ignoring the rest. It’s especially handy when you’re dealing with messy, high-dimensional data or when you want to build a simpler, more interpretable model without sacrificing accuracy. Think of it as a spotlight that highlights only the most relevant variables in your analysis.


* How did you test your model to determine if it is working reasonably correctly? *

=> Testing involved a mix of automated checks and real-world validation. For starters, I created small, controlled test cases to ensure the model handles basic scenarios—like refusing to run on empty data or flagging corrupted inputs. Then, I used real datasets to measure performance:

    ->Checked if predictions align closely with actual outcomes (e.g., plotting predicted vs. true values).
    ->Looked for "quiet" coefficients (ones near zero) in collinear datasets to confirm sparsity.
    ->Ran edge cases, like extreme regularization, to verify the model behaves intuitively (e.g., all coefficients vanishing if pushed too hard).
    ->Visualizations, like tracking how coefficients change as the model adjusts, also helped spot unexpected behavior.


* What parameters have you exposed to users of your implementation in order to tune performance? *

=> Users have two main levers to adjust:

    ->Regularization strength (λ): This acts like a "simplicity dial." Higher values force the model to prioritize fewer features, while lower values allow more complexity.
    ->Tolerance threshold: A behind-the-scenes setting that determines how small a coefficient must be to get trimmed to zero. Tightening this makes the model stricter about ignoring weak predictors.
    ->While the current version focuses on these, future updates could add auto-scaling for features or adaptive step sizes to handle tricky datasets.


* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental? *

=> The model struggles in two key scenarios:

    ->Noisy or unscaled data: If features aren’t standardized (e.g., age ranges vs. income in dollars), the model might overemphasize larger-scaled features. A quick fix would be adding auto-scaling.
    ->Extreme collinearity: When predictors are nearly identical (e.g., height in inches and centimeters), the model might randomly pick one instead of ignoring both.


**Extras**
* What should happen when you feed it highly collinear data? *

=> As *test_highly_collinear_data_sparsity*  this test indicates that the model correctly induces sparsity in the presence of highly collinear features—demonstrating that, among the three nearly identical features, only one (or at most two) has a significant coefficient. This behavior is exactly what we expect from a well-implemented LASSO model.
