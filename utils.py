from used_packages import *

def clean_title(title):
    return title.replace('\n', '-')

#%%

def glasso_paths(X_train, y_train, X_test, groups_dict_original=None, group_names_original=None, c_start=-6, c_stop=2, c_num=10, l1_reg=0.05,
                 scoring='neg_mean_squared_error', no_groups=False,
                 n_iter=100, tol=1e-5, cmap='nipy_spectral', title=[],
                 verbose=False, save_plot=False, show_nonzero_only=False):
    from sklearn.linear_model import Ridge
    from matplotlib.legend import Legend
    import math

    scaler_tr = StandardScaler()
    X_train_scaled = scaler_tr.fit_transform(X_train)
    scaler_ts = StandardScaler()
    X_test_scaled = scaler_ts.fit_transform(X_test)
    screr = get_scorer(scoring)
    count = 0

    if no_groups:
        # Create a list of sequential group numbers
        groups = list(range(1, len(X_train.columns) + 1))
        group_plot = groups  # Use the generated groups list directly
        labels = X_train.columns.tolist()
        groups_original = groups
        group_names_final = None
    else:
        groups_reset = []
        groups_original = []
        group_names_reset = {}
        group_names_final = {}
        track = []
        count1 = -1
        # obtain groups
        for col in X_train.columns:
            if groups_dict_original[col] not in track:
                count1 += 1
                track.append(groups_dict_original[col])
                group_names_reset[count1] = group_names_original[groups_dict_original[col]]
                group_names_final[groups_dict_original[col]] = group_names_original[groups_dict_original[col]]
            groups_reset.append(count1)
            groups_original.append(groups_dict_original[col])
        labels = list(group_names_reset.values())

        groups = groups_reset
        group_plot = groups.copy()

    if verbose:
        print(f'===============\nOriginal Labels\n===============\n{labels}\n')
        print(f'===============\nGroup Numbers\n===============\n{groups}\n')

    coefs = []
    scores = []
    lambdas = np.logspace(c_start, c_stop, c_num)
    best_lambda = None
    best_score = float(-np.inf)
    best_ypred = None
    best_glasso_coefs = None
    best_selected_features = None
    best_ridge_model = None

    np.random.seed(0)
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=0)
    print(X_tr.shape)

    for alpha in lambdas:
        count += 1
        # First use GroupLasso for feature selection
        group_lasso = GroupLasso(groups=groups, group_reg=alpha, l1_reg=l1_reg,
                                 n_iter=n_iter,
                                 tol=tol,
                                 scale_reg="inverse_group_size",
                                 subsampling_scheme=1, warm_start=True)
        group_lasso.fit(X_tr, y_tr)

        # Store the coefficients for plotting
        coefs.append(group_lasso.coef_)

        # Get selected features (non-zero coefficients)
        selected_features = np.abs(group_lasso.coef_) > 1e-10

        # If no features selected, continue to next lambda
        if not np.any(selected_features):
            scores.append(-np.inf)
            continue

        # Create Ridge model with selected features
        ridge_model = Ridge(alpha=1.0)

        X_tr_selected = X_tr[:, selected_features.flatten()]
        X_ts_selected = X_ts[:, selected_features.flatten()]

        ridge_model.fit(X_tr_selected, y_tr)

        # Score the Ridge model
        scr = screr(ridge_model, X_ts_selected, y_ts)

        if verbose:
            print(f'{count}/{c_num} lambda: {alpha:.2e} - MAPE = {-scr:.4f}, Selected features: {np.sum(selected_features)}')

        scores.append(scr)

        if scr > best_score:
            best_score = scr
            best_lambda = alpha
            # Store the best GroupLasso model and its coefficients
            best_glasso_model = group_lasso
            best_glasso_coefs = group_lasso.coef_.copy()  # Make a copy to ensure it's preserved
            best_selected_features = selected_features.flatten()

            # Store the best Ridge model
            best_ridge_model = ridge_model

    # Check if a valid model was found
    if best_lambda is None:
        raise ValueError("Best lambda not found.")

    # Calculate best_ypred
    X_test_selected = X_test_scaled[:, best_selected_features]
    best_ypred = best_ridge_model.predict(X_test_selected)

    # Add early return if only one feature is selected
    if np.sum(best_selected_features) == 1:
        return best_ridge_model, best_glasso_coefs, best_lambda, best_ypred, groups_original, group_names_final, scores, best_selected_features

    coefs = np.array(coefs)


    fig = plt.figure(figsize=(16, 8))

    # Filter features based on show_nonzero_only parameter
    if show_nonzero_only:
        nonzero_indices = np.where(np.abs(best_glasso_coefs) > 1e-10)[0]

        # Get the filtered labels and group plot
        filtered_group_plot = []
        filtered_labels = []

        for i in nonzero_indices:
            filtered_group_plot.append(group_plot[i] if i < len(group_plot) else i)

            # Handle the case where group_plot[i] might be out of range for labels
            if i < len(group_plot) and group_plot[i] < len(labels):
                filtered_labels.append(labels[group_plot[i]])
            else:
                # Add a fallback label
                filtered_labels.append(f"Feature {i}")

        # Mask for the coefficient matrix
        mask = np.zeros_like(coefs, dtype=bool)
        for i, idx in enumerate(nonzero_indices):
            mask[:, idx] = True
        filtered_coefs = np.where(mask, coefs, np.nan)
    else:
        filtered_labels = labels
        filtered_group_plot = group_plot
        filtered_coefs = coefs
        nonzero_indices = range(coefs.shape[1])

    # Number of features to display
    num_features = len(filtered_labels)

    # Determine legend columns based on number of features
    if num_features <= 30:
        legend_cols = 1
    elif num_features <= 60:
        legend_cols = 2
    else:
        legend_cols = 3

    # Create gridspec with two areas - one for plot, one for legend
    # Adjust width ratios based on number of legend columns
    legend_width_ratio = 0.6 * legend_cols
    gs = plt.GridSpec(1, 2, width_ratios=[3, legend_width_ratio], figure=fig)

    ax = fig.add_subplot(gs[0])

    colors = plt.cm.get_cmap((cmap), len(set(group_plot))+1)

    # Plot the coefficient paths
    for i, idx in enumerate(nonzero_indices):
        if show_nonzero_only:
            # Plot only non-zero coefficient paths
            if idx < len(group_plot):
                color_index = group_plot[idx]
                ax.plot(np.log(lambdas), coefs[:, idx], color=colors(color_index-1))
        else:
            # Plot all coefficient paths
            if idx < len(group_plot):
                color_index = group_plot[idx]
                ax.plot(np.log(lambdas), coefs[:, idx], color=colors(color_index-1))

    ax.axvline(np.log(best_lambda), linestyle='--', color='black', label='Optimal lambda')

    lim = ax.get_ylim()[0]
    lim_diff = np.abs(ax.get_ylim()[0] - ax.get_ylim()[1])
    x_lim_diff = np.abs(ax.get_xlim()[0] - ax.get_xlim()[1])

    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.9*lim_diff)), f'Opt. Lambda = {best_lambda:.2e}', color='black', ha='left', rotation=0)
    ax.text(np.log(best_lambda)+0.01*x_lim_diff, (lim+(0.85*lim_diff)), f'Opt. MAPE = {-best_score:.4f}', color='red', ha='left', rotation=0)

    ax.set_xlabel('Log Lambda')
    ax.set_ylabel('Scaled Coefficients')
    title_suffix = " (Non-zero coefficients only)" if show_nonzero_only else ""
    ax.set_title(f'{title}{title_suffix}')

    # Create proxy artists for the legend
    proxy_artists = []
    labels_to_show = []

    # Create mapping of group indices to labels
    group_to_label_map = {}

    if show_nonzero_only:
        # For non-zero features only
        for i, idx in enumerate(nonzero_indices):
            if idx < len(group_plot):
                group = group_plot[idx]
                if idx < len(X_train.columns):
                    # Use column name directly if available
                    label = X_train.columns[idx]
                else:
                    # Otherwise use group labels
                    label = filtered_labels[i] if i < len(filtered_labels) else f"Feature {idx}"
                group_to_label_map[group] = label
    else:
        # For all features
        for i, group in enumerate(group_plot):
            if i < len(labels):
                group_to_label_map[group] = labels[i]

    # Get unique groups that are actually used in the plot
    used_groups = set(filtered_group_plot)

    # Create proxy artists for each unique group
    for group in sorted(used_groups):
        proxy_artists.append(plt.Line2D([0], [0], color=colors(group-1), lw=2))
        # Find the corresponding label for this group
        label = group_to_label_map.get(group, f"Group {group}")
        labels_to_show.append(label)
    if not show_nonzero_only:
        labels_to_show = labels
    # Add legend in the right gridspec cell
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')  # Hide the axis

    # Get the height of the main plot area
    plot_height = ax.get_position().height * fig.get_figheight()

    # Calculate legend parameters
    font_size = 10
    line_height = 1.2
    item_height_inches = font_size * line_height / 72

    # Calculate max items per column based on plot height
    max_items_per_column = int(plot_height / item_height_inches)

    # Ensure we have at least one item per column
    items_per_column = max(1, min(max_items_per_column, len(labels_to_show)))

    # Recalculate number of columns based on items per column
    legend_cols = math.ceil(len(labels_to_show) / items_per_column)

    if no_groups:
        feat_title = 'Features'
        non_zero = 'Non-zero Features'
    else:
        feat_title = 'Groups'
        non_zero = 'Non-zero Groups'

    # Create the legend
    legend_title = f"{non_zero}" if show_nonzero_only else f"{feat_title}"
    legend = legend_ax.legend(proxy_artists, labels_to_show,
                              loc='upper left',
                              bbox_to_anchor=(0, 1),
                              ncol=legend_cols,
                              fontsize=font_size,
                              title=legend_title,
                              frameon=True,
                              title_fontsize=font_size+2)

    legend._legend_box.sep = 5
    legend._legend_box.pad = 5

    plt.tight_layout()

    if save_plot:
        nonzero_suffix = "_nonzero" if show_nonzero_only else ""
        title_save = clean_title(title)
        plt.savefig(f'output/{title_save}{nonzero_suffix}.pdf', bbox_inches='tight')
    plt.show()

    return best_ridge_model, best_glasso_coefs, best_lambda, best_ypred, groups_original, group_names_final, scores, best_selected_features



#%%

def coef_analysis(X, coefs, feature_names, group_names):
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': coefs.flatten(),
        'group': feature_names
    })

    # Group by the group number and analyze
    group_analysis = coef_df.groupby('group').agg({
        'coefficient': ['mean', 'sum', lambda x: np.sum(np.abs(x)), 'count', lambda x: np.sum(np.abs(x) > 1e-10), lambda x: np.mean(np.abs(x))],
        'feature': 'first'  # Just to get a sample feature name for reference
    })

    group_analysis.columns = ['mean_coef', 'sum_coef', 'abs_sum_coef', 'total_features', 'selected_features', 'mean_abs_coef', 'sample_feature']

    try:
        group_analysis['group_name'] = group_analysis.index.map(lambda x: group_names.get(x, f'Group {x}'))
    except:
        group_analysis['group_name'] = group_analysis.index.map(lambda x: X.columns[x-1])

    group_analysis['group'] = group_analysis.index

    group_analysis = group_analysis.reset_index(drop=True)

    for col in ['mean_coef', 'sum_coef', 'abs_sum_coef', 'mean_abs_coef']:
        group_analysis[col] = group_analysis[col].round(3)

    # Sort by absolute mean in descending order
    group_analysis = group_analysis.sort_values('mean_abs_coef', ascending=False).reset_index(drop=True)

    # Reorder columns
    cols = ['group_name', 'mean_abs_coef', 'mean_coef', 'total_features', 'selected_features', 'group']
    group_analysis = group_analysis[cols]

    return group_analysis

#%%

def xgboost_bayes(X, y, title='', save_plot=False):
    from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
    from bayes_opt import BayesianOptimization
    import scipy.stats as stats

    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)

    # Initialize the XGBoost regressor with default parameters
    default_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Evaluate baseline model with cross-validation
    cv_scores_default = cross_val_score(
        default_model, X, y,
        cv=kf,
        scoring='neg_mean_absolute_percentage_error'
    )

    mean_cv_score_default = -np.mean(cv_scores_default)  # Negate to get positive MAPE
    std_cv_score_default = np.std(cv_scores_default)

    print(f"Default Model CV MAPE: {mean_cv_score_default:.4f} ± {std_cv_score_default:.4f} (or {mean_cv_score_default * 100:.2f}% ± {std_cv_score_default * 100:.2f}%)")

    # function to be optimized
    def xgb_evaluate(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
        params = {
            'n_estimators': int(round(n_estimators)),
            'max_depth': int(round(max_depth)),
            'learning_rate': learning_rate,
            'subsample': max(min(subsample, 1), 0),
            'colsample_bytree': max(min(colsample_bytree, 1), 0),
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE
        }

        model = xgb.XGBRegressor(**params)

        cv_scores = cross_val_score(
            model, X, y,
            cv=kf,
            scoring='neg_mean_absolute_percentage_error'
        )

        return np.mean(cv_scores)

    # parameter bounds
    param_bounds = {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'subsample': (0.6, 1.0),
        'colsample_bytree': (0.6, 1.0)
    }

    optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=param_bounds,
        random_state=RANDOM_STATE,
        verbose=1
    )

    print("Starting Bayesian optimization...")
    optimizer.maximize(init_points=5, n_iter=20)

    # Get best parameters
    best_params = optimizer.max['params']
    print("\nBest parameters:")
    for param, value in best_params.items():
        if param in ['n_estimators', 'max_depth']:
            best_params[param] = int(round(value))
        print(f"{param}: {best_params[param]}")

    # Train model with best parameters
    best_model = xgb.XGBRegressor(
        n_estimators=int(round(best_params['n_estimators'])),
        max_depth=int(round(best_params['max_depth'])),
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    )

    # Evaluate best model with cross-validation
    cv_scores_best = cross_val_score(
        best_model, X, y,
        cv=kf,
        scoring='neg_mean_absolute_percentage_error'
    )

    mean_cv_score_best = -np.mean(cv_scores_best)
    std_cv_score_best = np.std(cv_scores_best)

    print(f"\nBest Model CV MAPE: {mean_cv_score_best:.4f} ± {std_cv_score_best:.4f} (or {mean_cv_score_best * 100:.2f}% ± {std_cv_score_best * 100:.2f}%)")

    improvement = (mean_cv_score_default - mean_cv_score_best) / mean_cv_score_default * 100
    print(f"Improvement over baseline: {improvement:.2f}%")

    y_pred_cv = cross_val_predict(best_model, X, y, cv=kf)

    best_model.fit(X, y)

    correlations = {}
    for col in X.columns:
        corr, _ = stats.pearsonr(X[col], y)
        correlations[col] = corr

    n = 20
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance,
        'Correlation': [correlations[feat] for feat in feature_names]
    })

    # Add sign from correlation to importance
    importance_df['Signed_Importance'] = importance_df['Importance'] * np.sign(importance_df['Correlation'])
    importance_df = importance_df.sort_values('Importance', ascending=False)

    top_feat = importance_df.head(n).reset_index(drop=True)

    print(f"\nTop {n} feature importances:")
    print(top_feat)

    residuals = y - y_pred_cv
    mape_by_sample = np.abs(residuals / y)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))

    # Plot the actual vs predicted values
    plt.subplot(2, 2, 1)
    plt.scatter(y, y_pred_cv, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Actual Crime Rate')
    plt.ylabel('Predicted Crime Rate')
    plt.title('Actual vs Predicted Values')
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    # Plot residuals
    plt.subplot(2, 2, 2)
    plt.scatter(y, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Crime Rate')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    # Plot distribution of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    # Plot MAPE by actual value
    plt.subplot(2, 2, 4)
    plt.scatter(y, mape_by_sample, alpha=0.5)
    plt.axhline(y=mean_cv_score_best, color='r', linestyle='--', label=f'Mean MAPE: {mean_cv_score_best:.4f}')
    plt.xlabel('Actual Crime Rate')
    plt.ylabel('Absolute Percentage Error')
    plt.title('Error Distribution by Crime Rate')
    plt.legend()
    plt.grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    plt.tight_layout()

    if save_plot:
        plt.savefig(f'output/xgboost_model_diagnostics - {title}.pdf', bbox_inches='tight')
    plt.show()

    return best_model, top_feat


#%%
def crime_rate_month(merged_data, data_csv_clean):
    base_merged = merged_data[['CA', 'GEOG', '2000_POP', '2010_POP', '2020_POP', 'TOT_POP']].copy()
    for year in range(2015, 2026):
        if year <= 2020:
            weight_2020 = (year - 2010) / 10
            weight_2010 = 1 - weight_2020
            base_merged[f'{year}_POP'] = (weight_2010 * base_merged['2010_POP']) + (weight_2020 * base_merged['2020_POP'])
        else:
            annual_growth_rate = (base_merged['2020_POP'] / base_merged['2010_POP']) ** (1/10) - 1
            years_after_2020 = year - 2020
            base_merged[f'{year}_POP'] = base_merged['2020_POP'] * (1 + annual_growth_rate) ** years_after_2020

    detailed_crime = data_csv_clean.groupby(['Community Area', 'Primary Type', 'Year', 'Month'], observed=True)['ID'].count().reset_index()
    detailed_crime.columns = ['CA', 'Crime_Type', 'Year', 'Month', 'Crime_Count']
    detailed_crime['CA'] = detailed_crime['CA'].astype('int')

    detailed_crime['Time_ID'] = detailed_crime['Year'].astype(str) + '-' + detailed_crime['Month'].astype(str).str.zfill(2)
    detailed_crime['Crime_Type'] = detailed_crime['Crime_Type'].str.replace(' ', '_').str.replace('/', '_')

    year_pop_map = {year: f'{year}_POP' for year in range(2015, 2026)}

    detailed_crime = pd.merge(detailed_crime, base_merged, on='CA', how='left')
    detailed_crime['Year_POP'] = detailed_crime.apply(lambda row: row[year_pop_map[row['Year']]], axis=1)

    detailed_crime['Crime_Rate'] = detailed_crime['Crime_Count'] / detailed_crime['Year_POP'] * 1000

    pop_cols = [i for i in detailed_crime.columns.tolist() if 'POP' in i]
    detailed_crime.drop(columns=pop_cols, inplace=True)

    return detailed_crime

#%%


def citywide_crime_rate_year(merged_data, data_csv_clean, crime_map):

    ca_col = 'GEOID' if 'GEOID' in merged_data.columns else 'CA'
    pop_cols_needed = [ca_col, '2010_POP', '2020_POP']

    base_merged = merged_data[pop_cols_needed].copy()

    for pop_col in ['2010_POP', '2020_POP']:
        if pop_col in base_merged.columns:
            base_merged[pop_col] = pd.to_numeric(base_merged[pop_col], errors='coerce')
        else:
            raise ValueError(f"Required population column '{pop_col}' not found in merged_data.")

    city_pop_2010 = base_merged['2010_POP'].sum()
    city_pop_2020 = base_merged['2020_POP'].sum()

    city_pop_estimates = {}
    epsilon_pop = 1e-6
    use_estimated_pop = False

    if pd.isna(city_pop_2010) or pd.isna(city_pop_2020) or city_pop_2010 <= 0:
        raise ValueError("Missing or invalid citywide 2010/2020 population totals. Cannot interpolate/extrapolate.")
    else:
        use_estimated_pop = True
        denominator_2010 = city_pop_2010 if city_pop_2010 > 0 else epsilon_pop
        annual_growth_rate = (city_pop_2020 / denominator_2010) ** (1/10.0) - 1

        for year in range(2015, 2026):
            if year <= 2020:
                weight_2020 = (year - 2010) / 10.0
                weight_2010 = 1.0 - weight_2020
                city_pop_estimates[year] = (weight_2010 * city_pop_2010) + (weight_2020 * city_pop_2020)
            else:
                years_after_2020 = year - 2020
                city_pop_estimates[year] = city_pop_2020 * (1 + annual_growth_rate) ** years_after_2020

    annual_crime = data_csv_clean.copy()
    annual_crime['Crime_Type'] = annual_crime['Primary Type'].str.replace(' ', '_').str.replace('/', '_')
    annual_crime['Crime_Category'] = annual_crime['Crime_Type'].map(crime_map)

    city_annual_crime_category = annual_crime.groupby(['Year', 'Crime_Category'], observed=False, dropna=False)['ID'].count().reset_index()
    city_annual_crime_category.columns = ['Year', 'Crime_Category', 'Crime_Count']

    valid_years = list(range(2015, 2026))
    city_annual_crime_category_filtered = city_annual_crime_category[city_annual_crime_category['Year'].isin(valid_years)].copy()

    if use_estimated_pop:
        city_annual_crime_category_filtered['Est_Population'] = city_annual_crime_category_filtered['Year'].map(city_pop_estimates)
    else:
        city_annual_crime_category_filtered['Est_Population'] = np.nan


    valid_pop = city_annual_crime_category_filtered['Est_Population'].notna() & (city_annual_crime_category_filtered['Est_Population'] > 0)
    city_annual_crime_category_filtered['Crime_Rate'] = np.nan
    epsilon_rate = 1e-9

    city_annual_crime_category_filtered.loc[valid_pop, 'Crime_Rate'] = (
            city_annual_crime_category_filtered.loc[valid_pop, 'Crime_Count'] /
            (city_annual_crime_category_filtered.loc[valid_pop, 'Est_Population'] + epsilon_rate) * 1000
    )

    final_cols = ['Year', 'Crime_Category', 'Crime_Count', 'Est_Population', 'Crime_Rate']
    citywide_aggregated_crime_year = city_annual_crime_category_filtered[final_cols]

    return citywide_aggregated_crime_year


#%%
def corr_pairs(df, features=None):
    if features is not None:
        corr_matrix = df[features].corr()
    else:
        corr_matrix = df.corr()

    high_corr_mask = (abs(corr_matrix) >= 0.5) & (abs(corr_matrix) < 1.0)  # Excluding self-correlations (which are always 1.0)

    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):  # only upper triangle to avoid duplicates
            if high_corr_mask.iloc[i, j]:
                high_corr_pairs.append((features[i], features[j], corr_matrix.iloc[i, j]))

    for feat1, feat2, corr_val in high_corr_pairs:
        print(f"{feat1} & {feat2}: {corr_val:.3f}")
    return high_corr_pairs