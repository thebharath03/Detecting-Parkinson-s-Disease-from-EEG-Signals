import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif,
    mutual_info_classif, chi2, RFE
)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ======== Parameters ========
CSV_FILE = "extracted_features_v2/features_all_subjects.csv"
TARGET_COLUMN = "label"
TEST_SIZE = 0.3
RANDOM_STATE = 42
N_FEATURES = 500
CV_FOLDS = 5
# ======== Load Data ========
print("Loading data...")
df = pd.read_csv(CSV_FILE)
exclude_cols = ['subject_id', TARGET_COLUMN, 'orig_label']
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(df.shape)
X = df[feature_cols].values
y = df[TARGET_COLUMN].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ======== Train-Test Split ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=TEST_SIZE, stratify=y_encoded, random_state=RANDOM_STATE
)

# ======== Scaling ========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======== Remove constant/near-constant features (common preprocessing) ========
selector_var = VarianceThreshold(threshold=1e-6)
X_train_var = selector_var.fit_transform(X_train_scaled)
X_test_var = selector_var.transform(X_test_scaled)

print(f"After variance threshold: {X_train_var.shape[1]} features\n")

# ======== Define Feature Selection Methods ========
feature_selection_methods = {
    "ANOVA F-test": SelectKBest(score_func=f_classif, k=min(N_FEATURES, X_train_var.shape[1])),
    "Mutual Information": SelectKBest(score_func=mutual_info_classif, k=min(N_FEATURES, X_train_var.shape[1])),
    "Chi-Square": SelectKBest(score_func=chi2, k=min(N_FEATURES, X_train_var.shape[1])),
    "RFE (LogReg)": RFE(
        estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        n_features_to_select=min(N_FEATURES, X_train_var.shape[1])
    )
}

# ======== Define Models ========
models = {
    "SVM_RBF": SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced'),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
}

# ======== Define scoring metrics ========
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(
        lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[0]),
    'recall': make_scorer(
        lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[1]),
    'f1': make_scorer(
        lambda y_true, y_pred: precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)[2])
}

# ======== Storage for Results ========
all_results = []

# ======== Train & Evaluate for Each Feature Selection Method ========
for fs_name, fs_method in feature_selection_methods.items():
    print(f"\n{'#' * 60}")
    print(f"Feature Selection Method: {fs_name}")
    print(f"{'#' * 60}\n")

    # Handle Chi-Square separately (requires non-negative features)
    if fs_name == "Chi-Square":
        # Make features non-negative by shifting to min=0
        X_train_fs = X_train_var - X_train_var.min() + 1e-10
        X_test_fs = X_test_var - X_test_var.min() + 1e-10
    else:
        X_train_fs = X_train_var
        X_test_fs = X_test_var

    # Apply feature selection
    X_train_selected = fs_method.fit_transform(X_train_fs, y_train)
    X_test_selected = fs_method.transform(X_test_fs)

    print(f"Selected {X_train_selected.shape[1]} features\n")

    # Store results for this feature selection method
    fs_results = []

    # Train each model
    for model_name, clf in models.items():
        print(f"Training {model_name} with {CV_FOLDS}-fold CV...")

        # Perform cross-validation
        cv_results = cross_validate(
            clf, X_train_selected, y_train,
            cv=CV_FOLDS,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Calculate mean CV scores
        cv_acc = cv_results['test_accuracy'].mean()
        cv_prec = cv_results['test_precision'].mean()
        cv_rec = cv_results['test_recall'].mean()
        cv_f1 = cv_results['test_f1'].mean()

        # Train on full training set and evaluate on test set
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)

        # Calculate test set metrics
        test_acc = accuracy_score(y_test, y_pred)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # Store results
        result = {
            'Feature Selection': fs_name,
            'Model': model_name,
            'CV_Accuracy': cv_acc,
            'CV_Precision': cv_prec,
            'CV_Recall': cv_rec,
            'CV_F1-Score': cv_f1,
            'Test_Accuracy': test_acc,
            'Test_Precision': test_prec,
            'Test_Recall': test_rec,
            'Test_F1-Score': test_f1
        }
        fs_results.append(result)
        all_results.append(result)

        print(f"  CV    - Acc: {cv_acc:.4f} | Prec: {cv_prec:.4f} | Rec: {cv_rec:.4f} | F1: {cv_f1:.4f}")
        print(f"  Test  - Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | F1: {test_f1:.4f}")

    # Display comparison table for this feature selection method
    print(f"\n{'-' * 80}")
    print(f"Results Summary for {fs_name}")
    print(f"{'-' * 80}")
    fs_df = pd.DataFrame(fs_results)
    fs_df_sorted = fs_df.sort_values('Test_F1-Score', ascending=False)
    print(fs_df_sorted.to_string(index=False))
    print()

# ======== Overall Comparison Table ========
print(f"\n{'=' * 100}")
print("OVERALL COMPARISON TABLE (Sorted by Test F1-Score)")
print(f"{'=' * 100}\n")

overall_df = pd.DataFrame(all_results)
overall_df_sorted = overall_df.sort_values('Test_F1-Score', ascending=False).reset_index(drop=True)
overall_df_sorted.index = overall_df_sorted.index + 1

# Format the output for better readability
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print(overall_df_sorted.to_string())

# ======== Save Results to CSV ========
overall_df_sorted.to_csv('model_comparison_results_v2.5.csv', index=False)
print(f"\n{'=' * 100}")
print("Results saved to 'model_comparison_results.csv'")
print(f"{'=' * 100}")

# ======== Summary Statistics ========
print(f"\n{'=' * 100}")
print("SUMMARY STATISTICS")
print(f"{'=' * 100}\n")

print("Best Performance by Feature Selection Method (Test F1-Score):")
best_by_fs = overall_df.loc[overall_df.groupby('Feature Selection')['Test_F1-Score'].idxmax()]
print(best_by_fs[['Feature Selection', 'Model', 'CV_F1-Score', 'Test_F1-Score']].to_string(index=False))

print("\nBest Performance by Model (Test F1-Score):")
best_by_model = overall_df.loc[overall_df.groupby('Model')['Test_F1-Score'].idxmax()]
print(best_by_model[['Model', 'Feature Selection', 'CV_F1-Score', 'Test_F1-Score']].to_string(index=False))

print(f"\nOverall Best Configuration (Based on Test F1-Score):")
best_config = overall_df_sorted.iloc[0]
print(f"  Feature Selection: {best_config['Feature Selection']}")
print(f"  Model: {best_config['Model']}")
print(f"  CV Accuracy: {best_config['CV_Accuracy']:.4f}")
print(f"  CV Precision: {best_config['CV_Precision']:.4f}")
print(f"  CV Recall: {best_config['CV_Recall']:.4f}")
print(f"  CV F1-Score: {best_config['CV_F1-Score']:.4f}")
print(f"  Test Accuracy: {best_config['Test_Accuracy']:.4f}")
print(f"  Test Precision: {best_config['Test_Precision']:.4f}")
print(f"  Test Recall: {best_config['Test_Recall']:.4f}")
print(f"  Test F1-Score: {best_config['Test_F1-Score']:.4f}")

# ======== Compare CV vs Test Performance ========
print(f"\n{'=' * 100}")
print("CV vs TEST PERFORMANCE ANALYSIS")
print(f"{'=' * 100}\n")

overall_df['F1_Difference'] = overall_df['CV_F1-Score'] - overall_df['Test_F1-Score']
overall_df['Acc_Difference'] = overall_df['CV_Accuracy'] - overall_df['Test_Accuracy']

print("Models with largest CV-Test F1 gap (potential overfitting):")
top_overfit = overall_df.nlargest(5, 'F1_Difference')[
    ['Feature Selection', 'Model', 'CV_F1-Score', 'Test_F1-Score', 'F1_Difference']]
print(top_overfit.to_string(index=False))

print("\nMost consistent models (smallest CV-Test F1 gap):")
most_consistent = overall_df.nsmallest(5, 'F1_Difference')[
    ['Feature Selection', 'Model', 'CV_F1-Score', 'Test_F1-Score', 'F1_Difference']]
print(most_consistent.to_string(index=False))

# ======== SHAP AND LIME ANALYSIS ========
print(f"\n{'=' * 100}")
print("SHAP AND LIME EXPLAINABILITY ANALYSIS")
print(f"{'=' * 100}\n")

# Use the best performing model configuration
best_fs_name = best_config['Feature Selection']
best_model_name = best_config['Model']

print(f"Performing explainability analysis on best model:")
print(f"  Feature Selection: {best_fs_name}")
print(f"  Model: {best_model_name}\n")

# Reconstruct the best model's data
if best_fs_name == "Chi-Square":
    X_train_fs = X_train_var - X_train_var.min() + 1e-10
    X_test_fs = X_test_var - X_test_var.min() + 1e-10
else:
    X_train_fs = X_train_var
    X_test_fs = X_test_var

# Apply the best feature selection method
best_fs_method = feature_selection_methods[best_fs_name]
X_train_best = best_fs_method.fit_transform(X_train_fs, y_train)
X_test_best = best_fs_method.transform(X_test_fs)

# Train the best model
best_model = models[best_model_name]
best_model.fit(X_train_best, y_train)

# Get selected feature indices for naming
selected_feature_mask = selector_var.get_support()
selected_features_after_var = [feature_cols[i] for i in range(len(feature_cols)) if selected_feature_mask[i]]

if hasattr(best_fs_method, 'get_support'):
    final_feature_mask = best_fs_method.get_support()
    selected_feature_names = [selected_features_after_var[i] for i in range(len(selected_features_after_var)) if
                              final_feature_mask[i]]
else:
    selected_feature_names = [f"Feature_{i}" for i in range(X_train_best.shape[1])]

# ======== SHAP Analysis ========
print("Generating SHAP explanations...")
print("Note: This may take several minutes depending on model complexity and data size.\n")

try:
    # Choose appropriate SHAP explainer based on model type
    if best_model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_best)

        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]  # Use positive class
        else:
            shap_values_to_plot = shap_values

    elif best_model_name == "LogisticRegression":
        explainer = shap.LinearExplainer(best_model, X_train_best)
        shap_values = explainer.shap_values(X_test_best)
        shap_values_to_plot = shap_values

    else:  # SVM_RBF, KNN - use KernelExplainer (slower)
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train_best, 100))
        shap_values = explainer.shap_values(X_test_best)

        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

    # SHAP Summary Plot - Global Feature Importance
    print("Creating SHAP summary plot (global feature importance)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_test_best,
                      feature_names=selected_feature_names,
                      show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    print("  Saved: shap_summary_plot.png\n")
    plt.close()

    # SHAP Bar Plot - Mean absolute SHAP values
    print("Creating SHAP bar plot (mean absolute importance)...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_to_plot, X_test_best,
                      feature_names=selected_feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
    print("  Saved: shap_bar_plot.png\n")
    plt.close()

    print("SHAP global analysis completed successfully.\n")

except Exception as e:
    print(f"Error during SHAP analysis: {str(e)}\n")
    shap_values_to_plot = None

# ======== Select Individual Samples for Explanation ========
print("Selecting individual samples for detailed explanation...")

# Find indices of PD and Control samples in test set
pd_indices = np.where(y_test == 1)[0]  # Assuming 1 is PD
control_indices = np.where(y_test == 0)[0]  # Assuming 0 is Control

if len(pd_indices) > 0 and len(control_indices) > 0:
    pd_idx = pd_indices[0]
    control_idx = control_indices[0]

    print(f"Selected PD patient: Test sample index {pd_idx}")
    print(f"Selected Control: Test sample index {control_idx}\n")

    # Get predictions for these samples
    pd_sample = X_test_best[pd_idx:pd_idx + 1]
    control_sample = X_test_best[control_idx:control_idx + 1]

    pd_pred = best_model.predict(pd_sample)[0]
    control_pred = best_model.predict(control_sample)[0]

    pd_pred_proba = best_model.predict_proba(pd_sample)[0] if hasattr(best_model, 'predict_proba') else None
    control_pred_proba = best_model.predict_proba(control_sample)[0] if hasattr(best_model, 'predict_proba') else None

    print(f"PD Sample - True Label: {le.inverse_transform([y_test[pd_idx]])[0]}, "
          f"Predicted: {le.inverse_transform([pd_pred])[0]}")
    if pd_pred_proba is not None:
        print(f"  Prediction Probabilities: {pd_pred_proba}")

    print(f"Control Sample - True Label: {le.inverse_transform([y_test[control_idx]])[0]}, "
          f"Predicted: {le.inverse_transform([control_pred])[0]}")
    if control_pred_proba is not None:
        print(f"  Prediction Probabilities: {control_pred_proba}\n")

    # ======== SHAP Force Plots for Individual Samples ========
    if shap_values_to_plot is not None:
        try:
            print("Creating SHAP force plots for individual samples...")

            # PD sample
            plt.figure(figsize=(20, 3))
            shap.force_plot(explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else
                            explainer.expected_value[1],
                            shap_values_to_plot[pd_idx],
                            X_test_best[pd_idx],
                            feature_names=selected_feature_names,
                            matplotlib=True, show=False)
            plt.title(f"SHAP Force Plot - PD Patient (Test Index {pd_idx})")
            plt.tight_layout()
            plt.savefig('shap_force_plot_pd.png', dpi=300, bbox_inches='tight')
            print("  Saved: shap_force_plot_pd.png")
            plt.close()

            # Control sample
            plt.figure(figsize=(20, 3))
            shap.force_plot(explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else
                            explainer.expected_value[1],
                            shap_values_to_plot[control_idx],
                            X_test_best[control_idx],
                            feature_names=selected_feature_names,
                            matplotlib=True, show=False)
            plt.title(f"SHAP Force Plot - Control (Test Index {control_idx})")
            plt.tight_layout()
            plt.savefig('shap_force_plot_control.png', dpi=300, bbox_inches='tight')
            print("  Saved: shap_force_plot_control.png\n")
            plt.close()

            # Waterfall plots
            print("Creating SHAP waterfall plots...")

            # PD sample waterfall
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap.Explanation(values=shap_values_to_plot[pd_idx],
                                                 base_values=explainer.expected_value if not isinstance(
                                                     explainer.expected_value, np.ndarray) else
                                                 explainer.expected_value[1],
                                                 data=X_test_best[pd_idx],
                                                 feature_names=selected_feature_names),
                                max_display=20, show=False)
            plt.title(f"SHAP Waterfall Plot - PD Patient (Test Index {pd_idx})")
            plt.tight_layout()
            plt.savefig('shap_waterfall_pd.png', dpi=300, bbox_inches='tight')
            print("  Saved: shap_waterfall_pd.png")
            plt.close()

            # Control sample waterfall
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap.Explanation(values=shap_values_to_plot[control_idx],
                                                 base_values=explainer.expected_value if not isinstance(
                                                     explainer.expected_value, np.ndarray) else
                                                 explainer.expected_value[1],
                                                 data=X_test_best[control_idx],
                                                 feature_names=selected_feature_names),
                                max_display=20, show=False)
            plt.title(f"SHAP Waterfall Plot - Control (Test Index {control_idx})")
            plt.tight_layout()
            plt.savefig('shap_waterfall_control.png', dpi=300, bbox_inches='tight')
            print("  Saved: shap_waterfall_control.png\n")
            plt.close()

        except Exception as e:
            print(f"Error creating SHAP individual plots: {str(e)}\n")

    # ======== LIME Analysis ========
    print("Generating LIME explanations...")

    try:
        # Initialize LIME explainer
        lime_explainer = LimeTabularExplainer(
            X_train_best,
            feature_names=selected_feature_names,
            class_names=le.classes_,
            mode='classification',
            random_state=RANDOM_STATE
        )

        # Explain PD sample
        print(f"Creating LIME explanation for PD patient...")
        lime_exp_pd = lime_explainer.explain_instance(
            X_test_best[pd_idx],
            best_model.predict_proba,
            num_features=20
        )

        fig = lime_exp_pd.as_pyplot_figure()
        plt.title(f"LIME Explanation - PD Patient (Test Index {pd_idx})")
        plt.tight_layout()
        plt.savefig('lime_explanation_pd.png', dpi=300, bbox_inches='tight')
        print("  Saved: lime_explanation_pd.png")
        plt.close()

        # Save text explanation
        with open('lime_explanation_pd.txt', 'w') as f:
            f.write(f"LIME Explanation for PD Patient (Test Index {pd_idx})\n")
            f.write(f"True Label: {le.inverse_transform([y_test[pd_idx]])[0]}\n")
            f.write(f"Predicted: {le.inverse_transform([pd_pred])[0]}\n")
            if pd_pred_proba is not None:
                f.write(f"Prediction Probabilities: {pd_pred_proba}\n\n")
            f.write(lime_exp_pd.as_list().__str__())
        print("  Saved: lime_explanation_pd.txt")

        # Explain Control sample
        print(f"Creating LIME explanation for Control...")
        lime_exp_control = lime_explainer.explain_instance(
            X_test_best[control_idx],
            best_model.predict_proba,
            num_features=20
        )

        fig = lime_exp_control.as_pyplot_figure()
        plt.title(f"LIME Explanation - Control (Test Index {control_idx})")
        plt.tight_layout()
        plt.savefig('lime_explanation_control.png', dpi=300, bbox_inches='tight')
        print("  Saved: lime_explanation_control.png")
        plt.close()

        # Save text explanation
        with open('lime_explanation_control.txt', 'w') as f:
            f.write(f"LIME Explanation for Control (Test Index {control_idx})\n")
            f.write(f"True Label: {le.inverse_transform([y_test[control_idx]])[0]}\n")
            f.write(f"Predicted: {le.inverse_transform([control_pred])[0]}\n")
            if control_pred_proba is not None:
                f.write(f"Prediction Probabilities: {control_pred_proba}\n\n")
            f.write(lime_exp_control.as_list().__str__())
        print("  Saved: lime_explanation_control.txt\n")

        print("LIME analysis completed successfully.\n")

    except Exception as e:
        print(f"Error during LIME analysis: {str(e)}\n")

else:
    print("Warning: Not enough PD or Control samples in test set for individual analysis.\n")

print(f"{'=' * 100}")
print("EXPLAINABILITY ANALYSIS COMPLETE")
print(f"{'=' * 100}")
print("\nGenerated files:")
print("  - shap_summary_plot.png: Global feature importance (beeswarm)")
print("  - shap_bar_plot.png: Mean absolute SHAP values")
print("  - shap_force_plot_pd.png: SHAP force plot for PD patient")
print("  - shap_force_plot_control.png: SHAP force plot for Control")
print("  - shap_waterfall_pd.png: SHAP waterfall plot for PD patient")
print("  - shap_waterfall_control.png: SHAP waterfall plot for Control")
print("  - lime_explanation_pd.png: LIME explanation for PD patient")
print("  - lime_explanation_pd.txt: LIME text explanation for PD patient")
print("  - lime_explanation_control.png: LIME explanation for Control")
print("  - lime_explanation_control.txt: LIME text explanation for Control")
print(f"{'=' * 100}")