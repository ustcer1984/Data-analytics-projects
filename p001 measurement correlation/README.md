# Common Mistakes in Measurement Machine Correlation

### Abstract

Machine correlation seems simple, which can be easily done with linear regression fitting. However, by doing virtual experiments using random numbers, we demonstrate the common mistakes such as wrong interpretation of $\small R^2$ score and system error from OLS linear regression model. Without confusing math derivation, we explain the concept using layman language and also recommend best practices for machine correlation job sample preperation and data processing.

### Summary

#### **Common mistakes in measurement machine correlation**

* $\small R^2$ score does NOT tell the correlation quality, i.e. the accuracy of the correlation coefficients.
* OLS linear regression which is commonly used for linear fitting generates system error (i.e. coefficients distribution mean to deviate from true values), due to variance of Machine X is ignored.

#### **Best practice recommendation**

* If you can only use **OLS** fitting due to software (e.g. `Excel`) limit.
  * Make ladder samples for correlation.
    * Sample range the larger the better, must be much larger than Machine X variance.
    * When total range is fixed, increase number of ladder steps does NOT improve correlation quality. 3 or 5 steps are enough.
    * With fixed range and step number, per step sample size is the larger the better.
    * If sample size is imbalanced among all steps, aggregate all data from same step into 1 point before fitting can reduce the system error induced by OLS linear regression model.
  * Avoid doing extrapolation prediction using the fitting result.
    * Meaning the correlation is only valid within sample range.
    * If you try to do prediction outside the sample range, the error may be high due to OLS linear model system error.
  * $\small R^2$ score may be higher with smaller sample size or aggregate fitting method. It does NOT mean better correlation quality or accuracy.
* If you can do **TLS** linear regression.
  * Correlation coefficients accuracy is less sensitive to sample range and distribution uniformity. Especially the distribution mean values shall always be close to true values.
    * There is less limit on doing extrapolation prediction, unlike OLS model.
  * TLS model is not robust to linear transformation of variables, you need to optimize their scales according to machine variance.
    * If difficult to decide the best scale, transform all variables to same scale (i.e. the correlation slope is close to 1) can usually yield reasonably good result.
    * Remember to apply the same variable rescaling when use the fitted model to do prediction in future.
