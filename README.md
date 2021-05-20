# DSAI_2021_HW02

## :camera_flash: Running my code
*   Virtual Python Environment
    *   If you use `virtualenv`, launch your environment and run `pip install -r requiements.txt`.
    *   If you use `pipenv`, run `pipenv install` after your environment is created.
*   Run main.py to output output.csv, some log info will show on your screen.
*   Example with using `pipenv`
```
$ pipenv run python main.py  # For Inference
$ pipenv run python train    # For Training
```


## 🗜️ Highlight of each phase in this repo.
*   Data Input Phase 
    *   Input training and testing csv data
*   Preprocessing Phase
    *   Encode time series data input and output for fitting model.
        *   Format: input last 7 days power consumption/generation, output next 1 day`s power consumption/generation
    *   Split encoded time series data into training set and validation set (80% / 20%)
*   Training Phase
    *   Utilize SVR by Scikit-learn, for power consumption/generation 
*   Infernece Phase
    *   Input testing data to model, call `model.predict()` to implement inferencing task.
    *   Do simple performance analysis by `RMSE`.
*   Postprocessing Phase
    *   **Based on inference result (Power Consumption/Generation), evaluate action of trading for next 1 days.**
* We use some Python modules to finish our work
    *   NumPy, Pandas, Scikit-Learn, Logru...etc. 

## ✈️Future work
*   For bidding, this work needs a estimator to ensure the optimal trading situation.
    *   Got last rank due to lack on cost management
*   For better bidding algorithm, we can utilize probability estimator to make maximize deal and minimize the cost.

# Reference
[SVR in Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
