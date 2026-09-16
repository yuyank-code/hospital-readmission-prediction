from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'data').mkdir(exist_ok=True); (ROOT/'models').mkdir(exist_ok=True); (ROOT/'results').mkdir(exist_ok=True)
rng=np.random.default_rng(42); n=30000
age=rng.integers(18,91,n); heart=rng.normal(78,13,n).clip(45,150); sbp=rng.normal(125,18,n).clip(75,210); glucose=rng.normal(125,45,n).clip(50,450); spo2=rng.normal(96.5,2,n).clip(80,100)
inpatient=rng.poisson(1.2,n); emergency=rng.poisson(1.7,n); outpatient=rng.poisson(2.5,n); los=rng.gamma(2.2,2.2,n).clip(1,30); diagnoses=rng.poisson(5,n).clip(1,16)
diag=rng.choice(['circulatory','diabetes','respiratory','renal','digestive','other'],n,p=[.24,.20,.18,.10,.10,.18]); gender=rng.choice(['Female','Male'],n); insulin=rng.choice(['No','Yes'],n,p=[.55,.45])
logit=-4.1+.018*age+.55*(diag=='circulatory')+.45*(diag=='renal')+.35*(diag=='respiratory')+.10*diagnoses+.18*inpatient+.11*emergency+.04*outpatient+.07*los+.004*(glucose-120)-.045*(spo2-95)+.006*(heart-75)+.002*(sbp-120)+.55*(insulin=='Yes')
p=1/(1+np.exp(-logit)); y=rng.binomial(1,p)
df=pd.DataFrame({'age':age,'heart_rate':heart,'systolic_bp':sbp,'glucose':glucose,'spo2':spo2,'prior_inpatient_visits':inpatient,'prior_emergency_visits':emergency,'prior_outpatient_visits':outpatient,'length_of_stay':los,'number_of_diagnoses':diagnoses,'diagnosis_category':diag,'gender':gender,'insulin':insulin,'readmitted_30d':y})
df.to_csv(ROOT/'data/synthetic_patient_records.csv',index=False)
X=df.drop(columns='readmitted_30d'); y=df.readmitted_30d
cat=['diagnosis_category','gender','insulin']; num=[c for c in X if c not in cat]
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
pipe=Pipeline([('preprocess',pre),('model',LogisticRegression(penalty='l2',C=1.0,class_weight='balanced',max_iter=2000,solver='liblinear',random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42); pipe.fit(Xtr,ytr); prob=pipe.predict_proba(Xte)[:,1]
auc=roc_auc_score(yte,prob); fpr,tpr,thresholds=roc_curve(yte,prob); i=np.argmin(abs(tpr-.80)); threshold=float(thresholds[i]); pred=(prob>=threshold).astype(int); cm=confusion_matrix(yte,pred)
metrics={'n_total':int(n),'positive_rate':float(y.mean()),'roc_auc':auc,'threshold_for_~80pct_recall':threshold,'precision_at_threshold':precision_score(yte,pred),'recall_at_threshold':recall_score(yte,pred),'f1_at_threshold':f1_score(yte,pred),'confusion_matrix':cm.tolist()}
joblib.dump(pipe,ROOT/'models/logistic_l2_readmission.joblib'); json.dump(metrics,open(ROOT/'results/metrics.json','w'),indent=2)
plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,label=f'ROC-AUC={auc:.3f}'); plt.plot([0,1],[0,1],'--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('30-Day Readmission ROC'); plt.legend(); plt.tight_layout(); plt.savefig(ROOT/'results/roc_curve.png',dpi=160); plt.close()
# Save coefficient ranking for interpretability.
feature_names=pipe.named_steps['preprocess'].get_feature_names_out(); coefs=pipe.named_steps['model'].coef_[0]
coef_df=pd.DataFrame({'feature':feature_names,'coefficient':coefs}); coef_df['abs_coefficient']=coef_df.coefficient.abs(); coef_df.sort_values('abs_coefficient',ascending=False).to_csv(ROOT/'results/top_coefficients.csv',index=False)
