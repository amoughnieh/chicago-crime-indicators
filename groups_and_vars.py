from used_packages import *

def group_names_and_variables():

    group_names = {
        0: "Geographic ID",
        1: "CA Name",
        2: "Total Pop. 2000",
        3: "Total Pop. 2010",
        4: "Total Pop. 2020",
        5: "Total Households 2020",
        6: "Avg. Household Size 2020",
        7: "Total Pop. 2022",
        8: "Age Cohort",
        9: "Race and Ethnicity",
        10: "Pop. living in households",
        11: "Pop. aged 16 and over",
        12: "Employment Status",
        13: "Mode of Travel to Work",
        14: "Aggregate travel time to work",
        15: "Vehicles Available",
        16: "Pop. aged 25 and over",
        17: "Education",
        18: "Household Income",
        19: "Household Occupancy",
        20: "Housing Type",
        21: "Housing Size",
        22: "Housing Age",
        23: "Home Value",
        24: "Rent",
        25: "Household PC & Net Access",
        26: "Disability No.",
        27: "Disability by Type",
        28: "Disability by Age",
        29: "Avg. Vehicles Miles",
        30: "Sales",
        31: "Equalized Assessed Value",
        32: "General Land Use",
        33: "Household Size",
        34: "Household Type",
        35: "Nativity",
        36: "Language"
    }

    groups_dict = {
        "GEOID": 0,
        "GEOG": 1,
        "2000_POP": 2,
        "2010_POP": 3,
        "2020_POP": 4,
        "2020_HH": 5,
        "2020_HH_SIZE": 6,
        "TOT_POP": 7,
        "UND5": 8, "A5_19": 8, "A20_34": 8, "A35_49": 8, "A50_64": 8, "A65_74": 8, "A75_84": 8, "OV85": 8, "MED_AGE": 8,
        "WHITE": 9, "HISP": 9, "BLACK": 9, "ASIAN": 9, "OTHER": 9,
        "POP_HH": 10,
        "POP_16OV": 11,
        "IN_LBFRC": 12, "EMP": 12, "UNEMP": 12, "NOT_IN_LBFRC": 12,
        "TOT_WRKR16OV": 13, "WORK_AT_HOME": 13, "TOT_COMM": 13, "DROVE_AL": 13, "CARPOOL": 13, "TRANSIT": 13, "WALK_BIKE": 13, "COMM_OTHER": 13,
        "AGG_TT": 14,
        "NO_VEH": 15, "ONE_VEH": 15, "TWO_VEH": 15, "THREEOM_VEH": 15,
        "POP_25OV": 16,
        "LT_HS": 17, "HS": 17, "SOME_COLL": 17, "ASSOC": 17, "BACH": 17, "GRAD_PROF": 17,
        "INC_LT_25K": 18, "INC_25_50K": 18, "INC_50_75K": 18, "INC_75_100K": 18, "INC_100_150K": 18, "INC_GT_150": 18, "MEDINC": 18, "INCPERCAP": 18,
        "TOT_HH": 19, "OWN_OCC_HU": 19, "RENT_OCC_HU": 19, "VAC_HU": 19,
        "HU_TOT": 20, "HU_SNG_DET": 20, "HU_SNG_ATT": 20, "HU_2UN": 20, "HU_3_4UN": 20, "HU_5_9UN": 20, "HU_10_19UN": 20, "HU_GT_19UN": 20, "HU_MOBILE": 20,
        "MED_ROOMS": 21, "BR_0_1": 21, "BR_2": 21, "BR_3": 21, "BR_4": 21, "BR_5": 21,
        "HA_AFT2010": 22, "HA_90_10": 22, "HA_70_90": 22, "HA_40_70": 22, "HA_BEF1940": 22, "MED_HA": 22,
        "HV_LT_150K": 23, "HV_150_300K": 23, "HV_300_500K": 23, "HV_GT_500K": 23, "MED_HV": 23,
        "CASHRENT_HH": 24, "RENT_LT500": 24, "RENT_500_999": 24, "RENT_1000_1499": 24, "RENT_1500_2499": 24, "RENT_GT2500": 24, "MED_RENT": 24,
        "COMPUTER": 25, "ONLY_SMARTPHONE": 25, "NO_COMPUTER": 25, "INTERNET": 25, "BROADBAND": 25, "NO_INTERNET": 25,
        "DISAB_ONE": 26, "DISAB_TWOMOR": 26, "DISAB_ANY": 26,
        "DIS_HEAR": 27, "DIS_VIS": 27, "DIS_COG": 27, "DIS_AMB": 27, "DIS_SLFCARE": 27, "DIS_INDPLIV": 27,
        "DIS_UND18": 28, "DIS_18_64": 28, "DIS_65_75": 28, "DIS_75OV": 28,
        "AVG_VMT": 29,
        "RET_SALES": 30, "GEN_MERCH": 30,
        "RES_EAV": 31, "CMRCL_EAV": 31, "IND_EAV": 31, "RAIL_EAV": 31, "FARM_EAV": 31, "MIN_EAV": 31, "TOT_EAV": 31,
        "TOT_ACRES": 32, "SF": 32, "Sfperc": 32, "MF": 32, "Mfperc": 32, "MIX": 32, "MIXperc": 32, "COMM": 32, "COMMperc": 32,
        "INST": 32, "INSTperc": 32, "IND": 32, "INDperc": 32, "TRANS": 32, "TRANSperc": 32, "AG": 32, "Agperc": 32, "OPEN": 32, "OPENperc": 32,
        "VACANT": 32, "VACperc": 32,
        "CT_1PHH": 33, "CT_2PHH": 33, "CT_3PHH": 33, "CT_4MPHH": 33,
        "CT_FAM_HH": 34, "CT_SP_WCHILD": 34, "CT_NONFAM_HH": 34,
        "NATIVE": 35, "FOR_BORN": 35,
        "NOT_ENGLISH": 36, "LING_ISO": 36, "ENGLISH": 36, "SPANISH": 36, "SLAVIC": 36, "CHINESE": 36, "TAGALOG": 36, "ARABIC": 36, "KOREAN": 36, "OTHER_ASIAN": 36, "OTHER_EURO": 36, "OTHER_UNSPEC": 36,
    }
    return groups_dict, group_names

def create_distilled_features(df):
    """
    Create distilled features from demographic data groups.

    Parameters:
    df (pandas.DataFrame): DataFrame containing all the original demographic features

    Returns:
    pandas.DataFrame: A DataFrame with distilled features
    """
    import numpy as np
    import pandas as pd

    """
    references:
    All ratios, rates, percentages, averages, or aggregates as defined in: American Community Survey and Puerto Rico Community Survey 2023 Subject Definitions (2023_ACSSubjectDefinitions.pdf)

    age_dependency: 2023_ACSSubjectDefinitions.pdf
    diversity_index: using Gini index, which is defined 2023_ACSSubjectDefinitions.pdf
    
    social_vulnerability_index: Social Vulnerability Index: A User’s Guide
    
    housing_age_diversity: ACS 2023
    housing_type_diversity: ACS 2023
    gentrification_risk: IDENTIFICATION OF INDICES TO QUANTIFY GENTRIFICATION
    
    Diversity indecis: An Introduction to Statistical Learning with Applications in Python (Gini Index)

    """

    # Create a new DataFrame for distilled features
    distilled_features = pd.DataFrame(index=df.index)

    # Population (groups 2-7)
    distilled_features['recent_population'] = df['TOT_POP'] if 'TOT_POP' in df.columns else df['2020_POP']
    distilled_features['avg_household_size'] = df['2020_HH_SIZE']

    # Age Cohort (8)
    distilled_features['median_age'] = df['MED_AGE']
    # Fix: Add small epsilon to prevent division by zero
    epsilon = 1e-10
    working_age_pop = df['A20_34'] + df['A35_49'] + df['A50_64']
    distilled_features['age_dependency'] = (df['UND5'] + df['OV85']) / (working_age_pop + epsilon)
    distilled_features['youth_ratio'] = df['UND5'] / (distilled_features['recent_population'] + epsilon)
    distilled_features['senior_ratio'] = (df['A65_74'] + df['A75_84'] + df['OV85']) / (distilled_features['recent_population'] + epsilon)

    # Race and Ethnicity (9)
    race_cols = ['WHITE', 'HISP', 'BLACK', 'ASIAN', 'OTHER']
    # Fix: Add small epsilon to prevent division by zero
    race_proportions = df[race_cols].div(df[race_cols].sum(axis=1) + epsilon, axis=0)
    # Diversity index (1 - sum of squared proportions) - higher means more diverse
    distilled_features['diversity_index'] = 1 - (race_proportions ** 2).sum(axis=1)
    # Largest demographic group percentage
    distilled_features['largest_demo_pct'] = df[race_cols].max(axis=1) / (df[race_cols].sum(axis=1) + epsilon)

    # Population in households (10)
    distilled_features['hh_population_ratio'] = df['POP_HH'] / (distilled_features['recent_population'] + epsilon)

    # Employment Status (12)
    distilled_features['employment_rate'] = df['EMP'] / (df['POP_16OV'] + epsilon)
    distilled_features['unemployment_rate'] = df['UNEMP'] / (df['IN_LBFRC'] + epsilon)
    distilled_features['labor_participation'] = df['IN_LBFRC'] / (df['POP_16OV'] + epsilon)

    # Travel to Work (13-14)
    distilled_features['transit_use_rate'] = df['TRANSIT'] / (df['TOT_COMM'] + epsilon)
    distilled_features['work_from_home_rate'] = df['WORK_AT_HOME'] / (df['TOT_WRKR16OV'] + epsilon)
    distilled_features['car_commute_rate'] = (df['DROVE_AL'] + df['CARPOOL']) / (df['TOT_COMM'] + epsilon)
    distilled_features['active_commute_rate'] = df['WALK_BIKE'] / (df['TOT_COMM'] + epsilon)
    # Average travel time per commuter
    distilled_features['avg_commute_time'] = df['AGG_TT'] / (df['TOT_COMM'] + epsilon)

    # Vehicles Available (15)
    distilled_features['car_ownership_rate'] = (df['ONE_VEH'] + df['TWO_VEH'] + df['THREEOM_VEH']) / (df['TOT_HH'] + epsilon)
    distilled_features['multi_car_rate'] = (df['TWO_VEH'] + df['THREEOM_VEH']) / (df['TOT_HH'] + epsilon)
    distilled_features['zero_car_rate'] = df['NO_VEH'] / (df['TOT_HH'] + epsilon)

    # Education (17)
    distilled_features['higher_education_rate'] = (df['BACH'] + df['GRAD_PROF']) / (df['POP_25OV'] + epsilon)
    distilled_features['hs_completion_rate'] = (df['HS'] + df['SOME_COLL'] + df['ASSOC'] + df['BACH'] + df['GRAD_PROF']) / (df['POP_25OV'] + epsilon)
    distilled_features['college_exposure_rate'] = (df['SOME_COLL'] + df['ASSOC'] + df['BACH'] + df['GRAD_PROF']) / (df['POP_25OV'] + epsilon)

    # Income (18)
    distilled_features['median_income'] = df['MEDINC']
    distilled_features['income_per_capita'] = df['INCPERCAP']
    distilled_features['high_income_pct'] = (df['INC_100_150K'] + df['INC_GT_150']) / (df['TOT_HH'] + epsilon)
    distilled_features['low_income_pct'] = df['INC_LT_25K'] / (df['TOT_HH'] + epsilon)

    # Housing Occupancy (19)
    distilled_features['homeownership_rate'] = df['OWN_OCC_HU'] / (df['TOT_HH'] + epsilon)
    distilled_features['rental_rate'] = df['RENT_OCC_HU'] / (df['TOT_HH'] + epsilon)
    distilled_features['vacancy_rate'] = df['VAC_HU'] / (df['HU_TOT'] + epsilon)

    # Housing Type (20)
    distilled_features['single_family_pct'] = (df['HU_SNG_DET'] + df['HU_SNG_ATT']) / (df['HU_TOT'] + epsilon)
    distilled_features['small_multifamily_pct'] = (df['HU_2UN'] + df['HU_3_4UN'] + df['HU_5_9UN']) / (df['HU_TOT'] + epsilon)
    distilled_features['large_multifamily_pct'] = (df['HU_10_19UN'] + df['HU_GT_19UN']) / (df['HU_TOT'] + epsilon)

    # Housing Size (21)
    distilled_features['median_rooms'] = df['MED_ROOMS']
    total_homes = df['BR_0_1'] + df['BR_2'] + df['BR_3'] + df['BR_4'] + df['BR_5'] + epsilon
    distilled_features['large_homes_pct'] = (df['BR_4'] + df['BR_5']) / total_homes
    distilled_features['small_homes_pct'] = df['BR_0_1'] / total_homes

    # Housing Age (22)
    distilled_features['median_home_age'] = df['MED_HA']
    distilled_features['new_housing_pct'] = (df['HA_AFT2010'] + df['HA_90_10']) / (df['HU_TOT'] + epsilon)
    distilled_features['old_housing_pct'] = df['HA_BEF1940'] / (df['HU_TOT'] + epsilon)

    # Home Value (23)
    distilled_features['median_home_value'] = df['MED_HV']
    distilled_features['high_value_homes_pct'] = (df['HV_300_500K'] + df['HV_GT_500K']) / ((df['HU_TOT'] - df['VAC_HU']) + epsilon)

    # Rent (24)
    distilled_features['median_rent'] = df['MED_RENT']
    distilled_features['high_rent_pct'] = (df['RENT_1500_2499'] + df['RENT_GT2500']) / (df['CASHRENT_HH'] + epsilon)
    distilled_features['rent_to_income'] = (df['MED_RENT'] * 12) / (df['MEDINC'] + epsilon)

    # Computer & Internet Access (25)
    distilled_features['internet_access_rate'] = df['INTERNET'] / (df['TOT_HH'] + epsilon)
    distilled_features['broadband_rate'] = df['BROADBAND'] / (df['TOT_HH'] + epsilon)
    distilled_features['digital_divide_rate'] = df['NO_COMPUTER'] / (df['TOT_HH'] + epsilon)

    # Disability (26-28)
    distilled_features['disability_rate'] = df['DISAB_ANY'] / (distilled_features['recent_population'] + epsilon)
    youth_pop = df['UND5'] + df['A5_19'] + epsilon
    distilled_features['youth_disability_rate'] = df['DIS_UND18'] / youth_pop
    senior_pop = df['A65_74'] + df['A75_84'] + df['OV85'] + epsilon
    distilled_features['senior_disability_rate'] = (df['DIS_65_75'] + df['DIS_75OV']) / senior_pop

    # Vehicle Miles (29)
    if 'AVG_VMT' in df.columns:
        distilled_features['avg_vehicle_miles'] = df['AVG_VMT']

    # Land Use (32)
    distilled_features['residential_land_pct'] = df['Sfperc'] + df['Mfperc']
    distilled_features['commercial_industrial_pct'] = df['COMMperc'] + df['INDperc']
    distilled_features['open_space_pct'] = df['OPENperc']
    distilled_features['vacant_land_pct'] = df['VACperc']

    # Household Size (33)
    distilled_features['single_person_hh_rate'] = df['CT_1PHH'] / (df['TOT_HH'] + epsilon)
    distilled_features['large_hh_rate'] = df['CT_4MPHH'] / (df['TOT_HH'] + epsilon)

    # Household Type (34)
    distilled_features['family_hh_rate'] = df['CT_FAM_HH'] / (df['TOT_HH'] + epsilon)
    distilled_features['single_parent_rate'] = df['CT_SP_WCHILD'] / (df['TOT_HH'] + epsilon)

    # Nativity (35)
    distilled_features['foreign_born_pct'] = df['FOR_BORN'] / (distilled_features['recent_population'] + epsilon)

    # Language (36)
    distilled_features['non_english_pct'] = df['NOT_ENGLISH'] / (distilled_features['recent_population'] + epsilon)
    distilled_features['linguistic_isolation_rate'] = df['LING_ISO'] / (df['TOT_HH'] + epsilon)

    # Cross-group relationships
    distilled_features['home_value_to_income'] = df['MED_HV'] / (df['MEDINC'] + epsilon)
    distilled_features['population_density'] = distilled_features['recent_population'] / (df['TOT_ACRES'] + epsilon)

    # Fix: Add error handling for optional columns
    if all(col in df.columns for col in ['2020_POP', '2010_POP']):
        distilled_features['pop_growth_rate_10yr'] = (df['2020_POP'] - df['2010_POP']) / (df['2010_POP'] + epsilon)
    else:
        distilled_features['pop_growth_rate_10yr'] = np.nan

    if all(col in df.columns for col in ['2020_POP', '2000_POP']):
        distilled_features['pop_growth_rate_20yr'] = (df['2020_POP'] - df['2000_POP']) / (df['2000_POP'] + epsilon)
    else:
        distilled_features['pop_growth_rate_20yr'] = np.nan

    if all(col in df.columns for col in ['TOT_POP', '2020_POP']):
        distilled_features['recent_growth_rate'] = (df['TOT_POP'] - df['2020_POP']) / (df['2020_POP'] + epsilon)
    else:
        distilled_features['recent_growth_rate'] = np.nan

    # Housing affordability metrics
    distilled_features['years_to_buy_median_home'] = df['MED_HV'] / (df['INCPERCAP'] + epsilon)
    distilled_features['rent_burden'] = (df['MED_RENT'] * 12) / ((df['MEDINC'] / 3) + epsilon)  # Rent as percentage of 1/3 of annual income

    # Age distribution metrics
    distilled_features['child_ratio'] = (df['UND5'] + df['A5_19']) / (distilled_features['recent_population'] + epsilon)
    distilled_features['working_age_ratio'] = (df['A20_34'] + df['A35_49'] + df['A50_64']) / (distilled_features['recent_population'] + epsilon)
    distilled_features['millennial_zoomer_ratio'] = df['A20_34'] / ((df['A35_49'] + df['A50_64']) + epsilon)

    # Mixed use and urban form metrics
    residential_commercial = df['Sfperc'] + df['Mfperc'] + df['COMMperc'] + df['INDperc'] + epsilon
    distilled_features['mixed_use_ratio'] = df['MIXperc'] / residential_commercial

    # helper function for min-max normalization
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series * 0 + 0.5  # Return a constant if all values are the same
        return (series - min_val) / (max_val - min_val)

    # Normalize the components that go into walkability proxy
    norm_res = normalize(distilled_features['residential_land_pct'])
    norm_pop_density = normalize(distilled_features['population_density'])
    norm_comm_ind = normalize(distilled_features['commercial_industrial_pct'])
    norm_active_commute = normalize(distilled_features['active_commute_rate'])

    # Geometric mean used because all components are essential for a good walkability score--if one is small, overall score should be small regardless of the other components--
    distilled_features['walkability_proxy'] = (norm_res * norm_pop_density * norm_comm_ind * norm_active_commute) ** 0.25

    # Social vulnerability metrics - normalize each component to 0-1 scale
    norm_disability = normalize(distilled_features['disability_rate'])
    norm_digital_divide = normalize(distilled_features['digital_divide_rate'])
    norm_ling_isolation = normalize(distilled_features['linguistic_isolation_rate'])
    norm_zero_car = normalize(distilled_features['zero_car_rate'])
    norm_low_income = normalize(distilled_features['low_income_pct'])

    distilled_features['social_vulnerability_index'] = (
                                                               norm_disability +
                                                               norm_digital_divide +
                                                               norm_ling_isolation +
                                                               norm_zero_car +
                                                               norm_low_income
                                                       ) / 5  # Average of normalized vulnerability indicators

    # Economic metrics - normalize each component to 0-1 scale
    norm_employment = normalize(distilled_features['employment_rate'])
    norm_higher_ed = normalize(distilled_features['higher_education_rate'])
    norm_high_income = normalize(distilled_features['high_income_pct'])
    norm_not_low_income = normalize(1 - distilled_features['low_income_pct'])

    distilled_features['economic_vitality_index'] = (
                                                            norm_employment +
                                                            norm_higher_ed +
                                                            norm_high_income +
                                                            norm_not_low_income
                                                    ) / 4  # Average of normalized economic indicators

    # Housing age diversity - ensure proper normalization
    housing_age_columns = ['HA_AFT2010', 'HA_90_10', 'HA_70_90', 'HA_40_70', 'HA_BEF1940']
    housing_age_props = df[housing_age_columns].div(df['HU_TOT'] + epsilon, axis=0)
    distilled_features['housing_age_diversity'] = 1 - (housing_age_props ** 2).sum(axis=1)

    # Housing type diversity - ensure proper normalization
    housing_type_props = pd.DataFrame({
        'single_family': distilled_features['single_family_pct'],
        'small_multifamily': distilled_features['small_multifamily_pct'],
        'large_multifamily': distilled_features['large_multifamily_pct'],
        'mobile': df['HU_MOBILE'] / (df['HU_TOT'] + epsilon)
    })
    distilled_features['housing_type_diversity'] = 1 - (housing_type_props ** 2).sum(axis=1)

    # Fix: Correctly create the components for gentrification risk
    # First, create the high/expensive rent percentage if it's missing
    if 'expensive_rent_pct' not in distilled_features.columns:
        distilled_features['expensive_rent_pct'] = distilled_features['high_rent_pct']

    # Create rent_to_med_income if it's missing
    if 'rent_to_med_income' not in distilled_features.columns:
        distilled_features['rent_to_med_income'] = distilled_features['rent_to_income']

    # Gentrification/displacement risk indicators - normalize components
    risk_components = ['rent_to_income', 'home_value_to_income', 'expensive_rent_pct',
                       'rent_to_med_income', 'new_housing_pct']

    # Create normalized versions of each component
    for component in risk_components:
        if component in distilled_features.columns:
            norm_name = f'{component}_norm'
            distilled_features[norm_name] = normalize(distilled_features[component])
        else:
            print(f"Warning: Component {component} not found in distilled_features")

    # Only include columns that exist in calculation
    existing_norm_components = [f'{c}_norm' for c in risk_components if f'{c}_norm' in distilled_features.columns]

    if len(existing_norm_components) > 0:
        # Weights normalized to sum to 1 based on available components
        weights = {
            'rent_to_income_norm': 0.3,
            'home_value_to_income_norm': 0.25,
            'expensive_rent_pct_norm': 0.2,
            'rent_to_med_income_norm': 0.15,
            'new_housing_pct_norm': 0.1
        }

        # Calculate weighted sum with available components
        total_weight = sum(weights[c] for c in existing_norm_components)
        distilled_features['gentrification_risk'] = sum(
            distilled_features[c] * (weights[c]/total_weight) for c in existing_norm_components
        )
    else:
        distilled_features['gentrification_risk'] = np.nan

    # Mobility dependence - normalize components before averaging
    norm_car_commute = normalize(distilled_features['car_commute_rate'])
    norm_multi_car = normalize(distilled_features['multi_car_rate'])
    norm_not_transit = normalize(1 - distilled_features['transit_use_rate'])
    norm_not_active = normalize(1 - distilled_features['active_commute_rate'])

    distilled_features['car_dependence_index'] = (
                                                         norm_car_commute +
                                                         norm_multi_car +
                                                         norm_not_transit +
                                                         norm_not_active
                                                 ) / 4  # Average of normalized car dependence indicators

    # Housing mismatch indicators
    # Calculate average bedrooms per unit
    total_br = (df['BR_0_1'] * 1.5 + df['BR_2'] * 2 + df['BR_3'] * 3 +
                df['BR_4'] * 4 + df['BR_5'] * 5)
    avg_br = total_br / (df['HU_TOT'] + epsilon)
    distilled_features['housing_size_mismatch'] = abs(
        distilled_features['avg_household_size'] - avg_br
    )

    # Family structure indicators
    distilled_features['non_traditional_hh_rate'] = (df['CT_NONFAM_HH'] + df['CT_SP_WCHILD']) / (df['TOT_HH'] + epsilon)

    # Assessed value per capita
    if 'TOT_EAV' in df.columns:
        distilled_features['eav_per_capita'] = df['TOT_EAV'] / (distilled_features['recent_population'] + epsilon)
        if 'RES_EAV' in df.columns:
            distilled_features['residential_eav_per_capita'] = df['RES_EAV'] / (distilled_features['recent_population'] + epsilon)
        if 'CMRCL_EAV' in df.columns:
            distilled_features['commercial_eav_per_capita'] = df['CMRCL_EAV'] / (distilled_features['recent_population'] + epsilon)

    # Segregation/integration indicators
    if all(col in df.columns for col in ['WHITE', 'BLACK', 'HISP', 'ASIAN']):
        max_group = df[['WHITE', 'BLACK', 'HISP', 'ASIAN']].max(axis=1)
        race_sum = df[['WHITE', 'BLACK', 'HISP', 'ASIAN']].sum(axis=1) + epsilon
        distilled_features['segregation_index'] = max_group / race_sum

    # Community resources - normalize components
    if 'INSTperc' in df.columns:
        distilled_features['institutional_land_ratio'] = df['INSTperc']

        # Normalize components for resource access index
        norm_inst_land = normalize(distilled_features['institutional_land_ratio'])
        norm_internet = normalize(distilled_features['internet_access_rate'])
        norm_car_access = normalize(1 - distilled_features['zero_car_rate'])
        norm_transit = normalize(distilled_features['transit_use_rate'])

        distilled_features['resource_access_index'] = (
                                                              norm_inst_land +
                                                              norm_internet +
                                                              norm_car_access +
                                                              norm_transit
                                                      ) / 4  # Average of normalized resource access indicators

    # Economic diversity
    income_cols = ['INC_LT_25K', 'INC_25_50K', 'INC_50_75K', 'INC_75_100K', 'INC_100_150K', 'INC_GT_150']
    income_props = df[income_cols].div(df['TOT_HH'] + epsilon, axis=0)
    distilled_features['income_diversity'] = 1 - (income_props ** 2).sum(axis=1)

    # Remove normalized components used for indices
    norm_columns = [col for col in distilled_features.columns if col.endswith('_norm')]
    distilled_features = distilled_features.drop(columns=norm_columns, errors='ignore')

    # Remove other intermediate or redundant features
    intermediate_features = [
        'expensive_rent_pct',
        'rent_to_med_income'
    ]
    distilled_features = distilled_features.drop(columns=intermediate_features, errors='ignore')

    # Handle any potential division by zero or missing values
    distilled_features = distilled_features.dropna(axis=1, how='any')

    return distilled_features



