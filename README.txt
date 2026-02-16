README - CS4375 Assignment 1, Part 1

Libraries Used:
- numpy       : numerical operations
- pandas      : data manipulation and CSV reading
- matplotlib  : plotting and saving figures
- scikit-learn: preprocessing (StandardScaler) and train/test split

Files Generated:
- training_log.txt                     : log of all iterations, learning rates, and MSE
- actual_vs_predicted_test.png         : scatter plot of actual vs predicted wine quality
- Residuals_vs_Predicted_Quality.png  : residuals plot

How to Run:
1. Make sure Python 3.x is installed.
2. Install required packages (preferably in a virtual environment):
   pip install numpy pandas matplotlib scikit-learn
3. Place the Python script in a folder.
4. Open a terminal or command prompt, navigate to the folder:
   cd path\to\folder
5. Run the script:
   python part1.py
6. The log file and plots will be generated in the same folder.

README — CS4375 Assignment 1, Part 2

Description:
This Python script (part2.py) performs linear regression using gradient descent
on the UCI Red Wine Quality dataset.



Libraries Used:
- numpy       
- pandas      
- matplotlib  
- scikit-learn

Files Generated:
- part2_log.txt           : log of all hyperparameter trials and best parameters
- actual_vs_predicted.png : scatter plot of actual vs predicted wine quality
- feature_coefficients.png: horizontal bar plot of feature coefficients

How to Run:
1. Make sure Python 3.x is installed.
2. Install required packages (preferably in a virtual environment):
   pip install numpy pandas matplotlib scikit-learn
3. Place part2.py in your desired folder.
4. Open a terminal or command prompt, navigate to the folder:
   cd path\to\folder
5. Run the script:
   python part2.py
6. The log file and plots will be generated in the same folder.

Notes:
- The script automatically downloads the dataset from the UCI repository.
- No local dataset files are needed.
- All file paths are relative, so it can run on any system with internet access
  and the required libraries.

