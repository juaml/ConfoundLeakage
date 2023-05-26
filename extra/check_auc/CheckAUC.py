#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import rpy2
import pandas as pd 
import numpy as np
from pathlib import Path
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[2]:


Path("./r_output/").mkdir(parents=True, exist_ok=True)


# In[3]:


get_ipython().run_cell_magic('R', '', 'library(\'glmnet\')\nfor (i in 1:100){\n    set.seed(i) \n    n <- 1000\n    df <- data.frame("x" = rep(c(0,1), n),\n    "y" = rbinom(n, 1, 0.25))\n    df$x_cr <- resid(lm(x~y, data=df))\n    \n    \n    df$y_hat_x <- predict(glm(y~x, data=df, family=\'binomial\'))\n    df$y_hat_x_cr <- predict(glm(y~x_cr, data=df, family=\'binomial\'))\n\n\n    library(MLmetrics)\n\n    auc_x = MLmetrics::AUC(df$x, df$y)\n    auc_x_cr = MLmetrics::AUC(df$x_cr, df$y)\n\n    df_scores = data.frame(auc_x, auc_x_cr ) \n    write.csv(df_scores, paste("./r_output/scores_",i, ".csv",  sep = ""))\n    write.csv(df, paste("./r_output/df_",i, ".csv",  sep = ""))\n}\n')


# In[4]:


df_source = pd.concat([pd.read_csv(f"./r_output/df_{i}.csv",index_col=0).assign(iteration=i)
                 for i in range(1,101)]
                )


df_scores = pd.concat([pd.read_csv(f"./r_output/scores_{i}.csv",index_col=0).assign(iteration=i)
                 for i in range(1,101)]
                )


# In[5]:


df_source.head()


# In[6]:


df_scores.head()


# In[7]:


df_train = df_source.drop(columns=["x_cr"])
df_train.head()


# In[8]:


def compute_aucs(df_train):
    confound_model = LinearRegression().fit(df_train[["y"]], df_train["x"])
    df_train["x_cr"] = df_train["x"] - confound_model.predict(df_train[["y"]])

    prediction_model = LogisticRegression(penalty="none", solver='newton-cg', class_weight=None ).fit(df_train[["x_cr"]], df_train["y"] )
    dt = DecisionTreeClassifier().fit(df_train[["x_cr"]], df_train["y"] )

    df_train["y_hat_x"] = prediction_model.predict_proba(df_train[["x"]])[:, 1]
    df_train["y_hat_x_cr"] = prediction_model.predict_proba(df_train[["x_cr"]])[:, 1]
    df_train["rf_y_hat_x"] = dt.predict_proba(df_train[["x"]])[:, 1]
    df_train["rf_y_hat_x_cr"] = dt.predict_proba(df_train[["x_cr"]])[:, 1]
    return pd.Series(
        dict(
        auc_x = roc_auc_score(df_train["y"], df_train["x"], ),
        auc_x_cr = roc_auc_score(df_train["y"], df_train["x_cr"], ),
        auc_y_hat_x = roc_auc_score(df_train["y"], df_train["y_hat_x"]),
        auc_y_hat_x_cr = roc_auc_score(df_train["y"], df_train["y_hat_x_cr"]),
        dt_auc_y_hat_x = roc_auc_score(df_train["y"], df_train["rf_y_hat_x"]),
        dt_auc_y_hat_x_cr = roc_auc_score(df_train["y"], df_train["rf_y_hat_x_cr"]) 
    ))


# In[9]:


df_train.head()


# In[10]:


df_scores_both = pd.concat(
    [
        df_scores.assign(program="R"),
        (df_train
         .drop(columns=['y_hat_x', 'y_hat_x_cr'])
         .groupby("iteration").apply(compute_aucs)
         .reset_index()
         .assign(program="Python")
        )
    ]
)
df_scores_both.head()


# In[11]:


df_scores_both_long = (df_scores_both
                        .drop(columns=["auc_y_hat_x", "auc_y_hat_x_cr"])
                        .melt(
                            value_vars=["auc_x","auc_x_cr"],
                            var_name="score",
                            id_vars=["program"]
                        )
        )
df_scores_both_long.head()


# In[12]:


ax = sns.violinplot(x="score",y="value", hue="program", 
              data=df_scores_both_long,split=True, inner="point")
ax.set_xticklabels(["AUC(X,y)", "AUC($X_{CR}$,y)"])
ax.set_xlabel("With or Without Confound Removed Features")
ax.set_ylabel("AUC")
ax.set_title("AUC behaves similary using MLmetrics or sklearn.metrics ")
ax.legend(loc="upper left")
ax.set_ylim(0,1)


# In[13]:


df_scores_python = (df_scores_both
                     .query("program=='Python'")
                     .melt(
                            value_vars=[
                                        "auc_y_hat_x", "auc_y_hat_x_cr", 
                                        "dt_auc_y_hat_x", "dt_auc_y_hat_x_cr"
                            ],
                            var_name="score",
                            id_vars=["program"]
                        )
                    
)


# In[14]:


df_scores_python.query('score  == "dt_auc_y_hat_x"').mean()


# In[15]:


ax = sns.boxenplot(x="score",y="value",  
              data=df_scores_python, )



ax.set_xticklabels(["AUC($\hat{y}_{lin}(X)$,y)", "AUC($\hat{y}_{lin}(X_{CR}$),y)",
                    "AUC($\hat{y}_{dt}(X)$,y)", "AUC($\hat{y}_{dt}(X_{CR}$),y)"

                   
                   
                   ])
ax.set_xlabel("AUC computation")
ax.set_ylabel("AUC")
ax.set_title("No Increase in AUC when actually predicting y")
ax.set_ylim(0,1.1)
ax.set_xticklabels(ax.get_xticklabels(),rotation=20)

