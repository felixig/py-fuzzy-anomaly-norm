# static combination of heterogeneous ensembles

import numpy as np
import pandas as pd
import sys
import os
import random

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from xgboost import XGBClassifier

import skfuzzy as fuzz
from sklearn.covariance import MinCovDet

from indices import get_indices
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
from scipy.special import erf

def normalize(s, method):
    if method=='abodreg':
        s = -1 * np.log10(s/np.max(s))
    if (method=='gauss' or method=='abodreg'):
        mu = np.nanmean(s)
        sigma = np.nanstd(s)
        s = (s - mu) / (sigma * np.sqrt(2))
        s = erf(s)
        s = s.clip(0, 1).ravel()
    elif method=='minmax':
        s = (s - s.min()) / (s.max() - s.min())
    return s


def bell_curve(x, a, b, c):
    return 1 / (1 + abs([x - c] / a) ** [2 * b])


def bell_params(x,method='fast'):
    c = np.quantile(x,0.5)
    a = np.quantile(x,0.9) - np.quantile(x,0.1)
    b = 2
    if method=='optimize':
        try:
            params, _ = curve_fit(bell_curve, x, y, p0=[a, b, c])
            a, b, c = params
        except:
            a, b, c = a, b, c

    return a,b,c


def gaussian_params(x,method):
    if method == 'normal':
        m = np.mean(x)
        s = np.std(x)
    elif method == 'mcv':
        try:
            mcd = MinCovDet().fit(x.reshape(-1, 1))
            m = mcd.location_[0]
            s = np.sqrt(mcd.covariance_[0, 0])
        except:
            print("Error in MCD! Switching to robust")
            m = np.median(x)
            s = robust_variation(x)        
    elif method == 'robust':
        m = np.median(x)
        s = robust_variation(x)        
    return m,s


def robust_variation(x):
    q25 = np.quantile(x,0.25)
    q75 = np.quantile(x,0.75)
    return (q75-q25)


def fuzz_phase(x, inliers_func='bell', outliers_func='s-shape'):

    mean, sigma = gaussian_params(x,'mcv')
    t = np.linspace(0,1,len(x))

    if inliers_func == 'gaussian':
        inlfunc = fuzz.gaussmf(x, mean, sigma)

    elif inliers_func == 'bell':
        a,b,c = bell_params(x,'optimize')
        inlfunc = fuzz.gbellmf(x, a, b, c)

    if outliers_func == 's-shape':
        fmax = np.max(x)
        outfunc = fuzz.smf(x, np.quantile(x,0.5), fmax)

    if outliers_func == 'trapez':
        p1,p2 = np.quantile(x,0.75),np.quantile(x,0.99)
        outfunc = fuzz.trapmf(x, [p1,p2,np.inf,np.inf])

    return inlfunc, outfunc


# ----------------- BEGIN -----------------

seed = 2024
random.seed(seed)
data_path  = sys.argv[1]

performance = []

# 1: experiment (1) - Anomaly Detection (unsupervised)
# P: probability normalization
# mA: Anomaly membership score
# 2: experiment (2) - Anomaly Modeling (supervised)
# M: Anomaly and Normmality membership degrees as two independent predictors
# auc: ROC-AUC score
# amf1: Adjusted Maximum F1 score
# aap: Adjusted Average Precision score
header = (['dataset', 
        '1Pauc', '1mAauc', '1Pamf1', '1mAamf1', '1Paap', '1mAaap', 
        '2Pauc_mean', '2Pauc_std', '2Mauc_mean', '2Mauc_std', 
        '2Pamf1_mean', '2Pamf1_std', '2Mamf1_mean', '2Mamf1_std', 
        '2Paap_mean', '2Paap_std', '2Maap_mean', '2Maap_std'])

# Iterate over datasets in [data_path]
for file_name in os.listdir(data_path):
    if file_name.endswith('.npz'):

        # ----------------- LOADING BENCHMARK DATASET -----------------
        data = np.load(data_path+file_name)
        X, y = data['X'], data['y']
        print(file_name,X.shape,y.shape,sum(y)/X.shape[0])


        # ----------------- OBTAINING ANOMALY SCORES (UNSUP.) -----------------

        # Initialize individual models
        model_iforest = IForest(random_state=seed)
        model_knn = KNN()
        model_lof = LOF()

        print("training...")
        # Train each model
        model_iforest.fit(X)
        model_knn.fit(X)
        model_lof.fit(X)

        # Get raw-scores 
        scores_iforest = model_iforest.decision_scores_
        scores_knn = model_knn.decision_scores_
        scores_lof = model_lof.decision_scores_  

        # Normalize based on fuzzy-sets 
        print("fuzzy-based normalization...")
        memi_ifs,memo_ifs = fuzz_phase(scores_iforest)
        memi_knn,memo_knn = fuzz_phase(scores_knn)
        memi_lof,memo_lof = fuzz_phase(scores_lof)

        # Normalize based on probabilistic approach 
        print("probabilistic normalization...")
        unif_scores_iforest = normalize(scores_iforest, 'gauss')
        unif_scores_knn = normalize(scores_knn, 'gauss')
        unif_scores_lof = normalize(scores_lof, 'gauss')

        # Heterogeneous Ensemble (simple averaging)
        print("ensembling...")
        ensemble_scores = (unif_scores_iforest + unif_scores_knn + unif_scores_lof) / 3     # proba
        ensemble_memo = (memo_ifs + memo_knn + memo_lof) / 3                                # memA
        ensemble_memi = (memi_ifs + memi_knn + memi_lof) / 3                                # memN

        # Get external validation indices
        print("estimating performances...")
        UNIF = get_indices(y,ensemble_scores)
        MEMB_out = get_indices(y,ensemble_memo)


        # ----------------- OBTAINING ANOMALY MODELING (SUP.) -----------------

        # Compute the positive class weight
        pos_class_weight = (len(y) - np.sum(y)) / np.sum(y)

        # Initialize XGBClassifier with scale_pos_weight
        print("supervised k-fold classification...")
        model = XGBClassifier(n_estimators=100, objective='binary:logistic', scale_pos_weight=pos_class_weight, max_delta_step=1, random_state=seed)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


        # MEMBERSHIP DEGREES -----------------

        # Anomaly and Normality membership degress as predictors {memA,memN}
        Xmemb = np.column_stack((ensemble_memo, ensemble_memi))

        # Evaluate each training/test-split fold 
        auc_kf, amf1_kf, aap_kf = [],[],[]
        for train_index, test_index in kf.split(Xmemb,y):

            X_train, X_test = Xmemb[train_index], Xmemb[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred_memb = model.predict(X_test)
            SMEMB = get_indices(y_test,pred_memb)
            auc_kf.append(SMEMB['auc']), amf1_kf.append(SMEMB['adj_maxf1']), aap_kf.append(SMEMB['adj_ap'])

        # Merge k-fold evaluations 
        SMEMB_auc_mean, SMEMB_auc_std = np.nanmean(auc_kf), np.nanstd(auc_kf)
        SMEMB_amf1_mean, SMEMB_amf1_std = np.nanmean(amf1_kf), np.nanstd(amf1_kf)
        SMEMB_aap_mean, SMEMB_aap_std = np.nanmean(aap_kf), np.nanstd(aap_kf)


        # PROBABILITIES -----------------

        # probability as predictor {proba}
        Xunif = ensemble_scores.reshape(-1,1)

        # Evaluate each training/test-split fold 
        auc_kf, amf1_kf, aap_kf = [],[],[]
        for train_index, test_index in kf.split(Xunif,y):

            X_train, X_test = Xunif[train_index], Xunif[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            pred_unif = model.predict(X_test)
            SUNIF = get_indices(y_test,pred_unif)
            auc_kf.append(SUNIF['auc']), amf1_kf.append(SUNIF['adj_maxf1']), aap_kf.append(SUNIF['adj_ap'])

        # Merge k-fold evaluations 
        SUNIF_auc_mean, SUNIF_auc_std = np.nanmean(auc_kf), np.nanstd(auc_kf)
        SUNIF_amf1_mean, SUNIF_amf1_std = np.nanmean(amf1_kf), np.nanstd(amf1_kf)
        SUNIF_aap_mean, SUNIF_aap_std = np.nanmean(aap_kf), np.nanstd(aap_kf)



        # ----------------- STORE RESULTS -----------------

        row = ([file_name, UNIF['auc'], MEMB_out['auc'], 
                UNIF['adj_maxf1'], MEMB_out['adj_maxf1'], 
                UNIF['adj_ap'], MEMB_out['adj_ap'], 
                SUNIF_auc_mean, SUNIF_auc_std, SMEMB_auc_mean, SMEMB_auc_std, 
                SUNIF_amf1_mean, SUNIF_amf1_std, SMEMB_amf1_mean, SMEMB_amf1_std,
                SUNIF_aap_mean, SUNIF_aap_std, SMEMB_aap_mean, SMEMB_aap_std])

        performance.append(row)

        df = pd.DataFrame(performance, columns=header).to_csv('perf.csv')

df = pd.DataFrame(performance, columns=header).to_csv('perf.csv')

