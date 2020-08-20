## Creditcard Fraud Detection (NearMiss & SMOTE)Algorithm on Imbalanced Dataset:
This is an excerpt from a notebook on Kaggle done by Janio Martinez.

### Our Goals:
Understand the little distribution of the "little" data that was provided to us.
Create a 50/50 sub-dataframe ratio of "Fraud" and "Non-Fraud" transactions. (NearMiss Algorithm)
Determine the Classifiers we are going to use and decide which one has a higher accuracy.
Create a Neural Network and compare the accuracy to our best classifier.
Understand common mistaked made with imbalanced datasets.

### Correcting Previous Mistakes from Imbalanced Datasets:
Never test on the oversampled or undersampled dataset.
If we want to implement cross validation, remember to oversample or undersample your training data during cross-validation, not before!
Don't use accuracy score as a metric with imbalanced datasets (will be usually high and misleading), instead use f1-score, precision/recall score or confusion matrix.

### Algorithm used:
#### NearMiss Algorithm (Under Sampling) :
* It aims to balance class distribution by randomly eliminating majority class examples. When instances of two different classes are very close to each other, we remove the instances of the majority class to increase the spaces between the two classes. This helps in the classification process.
* To prevent problem of information loss in most under-sampling techniques, near-neighbor methods are widely used.
#### Understanding SMOTE:(Over Sampling)
* Solving the Class Imbalance: SMOTE creates synthetic points from the minority class in order to reach an equal balance between the minority and majority class.
* Location of the synthetic points: SMOTE picks the distance between the closest neighbors of the minority class, in between these distances it creates synthetic points.
*Final Effect: More information is retained since we didn't have to delete any rows unlike in random undersampling.
* Accuracy || Time Tradeoff: Although it is likely that SMOTE will be more accurate than random under-sampling, it will take more time to train since no rows are eliminated as previously stated.

### Summary:
The transaction amount is relatively small. The mean of all the mounts made is approximately USD 88.
There are no "Null" values, so we don't have to work on ways to replace values.
Most of the transactions were Non-Fraud (99.83%) of the time, while Fraud transactions occurs (017%) of the time in the dataframe.

### Feature Technicalities:
##### Dimentionality Reduction:

*TSNE Transformation:
t-SNE algorithm can pretty accurately cluster the cases that were fraud and non-fraud in our dataset.
Although the subsample is pretty small, the t-SNE algorithm is able to detect clusters pretty accurately in every scenario (I shuffle the dataset before running t-SNE)
This gives us an indication that further predictive models will perform pretty well in separating fraud cases from non-fraud cases.

* PCA Transformation: The description of the data says that all the features went through a PCA transformation (Dimensionality Reduction technique) (Except for time and amount).

Scaling: Keep in mind that in order to implement a PCA transformation features need to be previously scaled. (In this case, all the V features have been scaled or at least that is what we are assuming the people that develop the dataset did.)

T-SNE took 7.0 s
PCA took 0.032 s
Truncated SVD took 0.0051 s

![alt text](https://www.kaggleusercontent.com/kf/16695845/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..gfsNHHJ53YZyIYQuQhQ_dg.JxikW66yo6orD364IvLgpHllmPlgwS6ACzTmOunVnMmSA4z8rfcY_raXItKZO75TUlEVhVerYkPG8M20L2PyFzoIC-kJROAqInYWw0rM-VQd5Lq_j89cjkCHBMVRmAwWUJZlVFlYhshnm2Dmd2-U72cRU8m-FXoy63HqHE2I3FApHpqurffTEAzP8QRpsZXIqZU0ANed4J40mFACprklRTOWQliOwJxGjjos_B7FRIjD9EQdIiz-O56kD0NieKvP-Q9pncWcwqkp3bkbRiqPPbxZHYcuEegPC3R7kwszlzcXc0ssnumR5DWHLYAtaEesgHFY2LOz1KDoLWR-UjdiFnip5lAVM-bVqWUFDjlQGD-Nsrz9f_7_OLud9HzWterjOmdhfXW_Lttb9S0tHAkHYOL7knSRX5ckJa7rnRda3dU73AjoIYwqsRV4icWBg0moJxoKAjyIGAsTo1KJSuqMWF7xqSAnrSRkJSInKrWCC_MHdxr6oCxFaQaGrahEiSTyo4PfbOY6z8ZosaBkgWtTVsl52UUdS54h8L5Y_h4QrRhuX1wovUbnNFqP6qyy_rLPbIWOcR_Oz4V78Fo-ArlkE7RxsXqsiOvK6rOkiQudJLWcJtnGKuNl_mlkTA-T0vsYZR80mA2bZZb1pXXbcp6hPnFd1r_RoRD8pXVA35SPaRGzN09RCTgRHzgIyA9JqvG4.bVUN-OEbF57w3MdbXu2-UA/__results___files/__results___30_0.png)

### Steps Involves: 
* Data Preprocessing,
* EDA,
* Dimentionality Reduction (PCA,TSNE, Truncated SVD),
* Undersamping (Near Miss Algorithm), OverSampling(SMOTE)

### Conclusion:
Implementing SMOTE on our imbalanced dataset helped us with the imbalance of our labels (more no fraud than fraud transactions). Nevertheless, I still have to state that sometimes the neural network on the oversampled dataset predicts less correct fraud transactions than our model using the undersample dataset. However, remember that the removal of outliers was implemented only on the random undersample dataset and not on the oversampled one. Also, in our undersample data our model is unable to detect for a large number of cases non fraud transactions correctly and instead, misclassifies those non fraud transactions as fraud cases. Imagine that people that were making regular purchases got their card blocked due to the reason that our model classified that transaction as a fraud transaction, this will be a huge disadvantage for the financial institution. The number of customer complaints and customer disatisfaction will increase. The next step of this analysis will be to do an outlier removal on our oversample dataset and see if our accuracy in the test set improves.























