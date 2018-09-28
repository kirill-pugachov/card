# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:27:06 2018

@author: User
"""


import pandas as pd
import numpy as np
#import time
import datetime
import csv
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from sklearn import ensemble
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

FILE_NAME = 'train_ds.csv'
FILE_PATH = 'C:\\Users\\tom\\card_garbage\\Data\\'
TEST_FILE_NAME = 'test_ds.csv'
CITIES_FILE_NAME = 'cities_list.csv'


def get_mcc_coding_list():
    avia = list(range(3000,3301))
    veterinary = [742]
    agricultural = [763]
    general_contractor = [1520]
    carpentry_contractors = [1750]
    miscellaneous_publishing_and_printing = [2741]
    car_rental_agencies = list(range(3351,3441))
    hotels_motels_resorts = list(range(3501,3836))
    railroads = [4011]
    commuter_passenger_transportation = [4111]
    passenger_railways = [4112]
    ambulance_services = [4119]
    taxicabs_limousines = [4121]
    bus_lines = [4131]
    motor_freight_carriers_and_trucking = [4214]
    courier_services = [4215]
    public_warehousing_and_storage = [4225]
    steamship_and_cruise_lines = [4411]
    boat_rentals_and_leasing = [4457]
    marinas_marine_service = [4468]
    airlines_and_air_carriers = [4511]
    airports_flying_fields_and_airport_terminals = [4582]
    travel_agencies = [4722]
    tolls_and_bridge_fees = [4784]
    transportation_services = [4789]
    telecommunication_equipment = [4812]
    telecommunication_services = [4814]
    computer_network = [4816]
    money_transfer = [4829]
    cable_satellite_and_other_pay_television = [4899]
    utilities_electric_gas_water_and_sanitary = [4900]
    motor_vehicle_supplies_and_new_parts = [5013]
    office_and_commercial_furniture = [5021]
    manual_cash_disbursements = [6010]
    automated_cash_disbursements = [6011]
    grocery_stores_and_supermarkets = [5411]
    miscellaneous_food_stores = [5499]
    eating_places_and_restaurants = [5812]
    candy_nut_and_Confectionery_stores = [5541]
    drug_stores_and_pharmacies = [5912]
    merchandise_services_and_debt_repayment = [6012]
    mastercard_moneysend = list(range(6536,6538))
    fast_food_restaurants = [5814]
    cosmetic_stores = [5977]
    family_clothing_stores = [5651]
    lumber_building_materials_stores = [5211]
    variety_stores = [5331]
    hotels_motels_and_resorts = [7011]
    men_s_women_s_clothing_stores = [5691]
    drinking_places = [5813]
    shoe_stores = [5661]
    confectionery = [5441]
    miscellaneous_business_services = [7399]
    motion_picture_theaters_cinema = [7832]
    betting_casino_gambling = [7995]
    department_stores = [5311]
    direct_marketing_merchant = [5968]
    children_s_and_infants_wear_stores = [5641]
    non_financial_institutions = [6051]
    household_appliance_stores = [5722]
    direct_marketing_catalog_merchant = [5964]
    sporting_goods_stores = [5941]
    duty_free_stores = [5309]
    hobby_toy_and_game_shops = [5945]
    sports_and_riding_apparel_stores = [5655]
    beauty_and_barber_shops = [7230]
    women_s_accessory_and_specialty_shops = [5631]
    package_stores_beer_wine_and_liquor = [5921]
    women_s_ready_to_wear_stores = [5621]
    professional_services = [8999]
    miscellaneous_and_specialty_retail_shops = [5999]
    book_stores = [5942]
    advertising_services = [7311]
    video_game_arcades_establishments = [7994]
    electronics_stores = [5732]
    chemicals_and_allied_products = [5169]
    membership_clubs_sports_recreation_athletic = [7997]
    gift_card_novelty_and_souvenir_shops = [5947]
    jewelry_stores_watches_clocks_and_silverware_stores = [5944]
    medical_services_and_health_practitioners = [8099]
    miscellaneous_apparel_and_accessory_shops = [5699]
    ticket_agencies_and_theatrical_producers = [7922]
    pet_shops_pet_foods_and_supplies_stores = [5995]
    furniture_home_furnishings_and_equipment_stores = [5712]
    men_s_and_boys_clothing_and_accessories_stores = [5611]
    miscellaneous_general_merchandise = [5399]
    health_and_beauty_spas = [7298]
    home_supply_warehouse_stores = [5200]
    record_stores = [5735]
    recreation_services = [7999]
    parking_lots_parking_meters_and_garages = [7523]
    used_merchandise_and_secondhand_stores = [5931]
    government_services = [9399]
    US_federal_government_agencies_or_departments = [9405]
    computer_software_stores = [5734]
    stationery_stores_office_and_school_supply_stores = [5943]
    florists_supplies_nursery_stock_and_flowers = [5193]
    freezer_and_locker_meat_provisioners = [5422]
    laundry_cleaning_and_garment_services = [7210]
    florists = [5992]
    opticians_optical_goods_and_eyeglasses = [8043]
    automotive_parts_and_accessories_stores = [5533]
    medical_and_dental_laboratories = [8071]
    car_rental_agencies = [7512]
    cigar_stores_and_stands = [5993]
    commercial_sports_professional_sports_clubs = [7941]
    tourist_attractions_and_exhibits = [7991]
    dairy_products_stores = [5451]
    miscellaneous_personal_services = [7299]
    glassware_crystal_stores = [5950]
    miscellaneous_home_furnishing_specialty_stores = [5719]
    luggage_and_leather_goods_stores = [5948]
    sewing_needlework_fabric_and_piece_goods_stores = [5949]
    information_retrieval_services = [7375]
    motion_picture_and_video_tape_production_and_distribution = [7829]
    dentists_and_orthodontists = [8021]
    computers_and_computer_peripheral_equipment_and_software = [5045]
    fuel_dealers_fuel_oil_wood_coal_and_liquefied_petroleum = [5983]
    bakeries_0 = [5462]
    charitable_social_service_organizations = [8398]


    mcc_coding_list = (avia, veterinary, agricultural, general_contractor, carpentry_contractors, miscellaneous_publishing_and_printing, 
    car_rental_agencies, hotels_motels_resorts, railroads, commuter_passenger_transportation, passenger_railways,
    ambulance_services, taxicabs_limousines, bus_lines, motor_freight_carriers_and_trucking, courier_services, 
    public_warehousing_and_storage, steamship_and_cruise_lines, boat_rentals_and_leasing, marinas_marine_service, 
    airlines_and_air_carriers, airports_flying_fields_and_airport_terminals, travel_agencies, tolls_and_bridge_fees,
    transportation_services, telecommunication_equipment, telecommunication_services, computer_network, money_transfer,
    cable_satellite_and_other_pay_television, utilities_electric_gas_water_and_sanitary, motor_vehicle_supplies_and_new_parts,
    office_and_commercial_furniture, manual_cash_disbursements, automated_cash_disbursements, grocery_stores_and_supermarkets,
    miscellaneous_food_stores, eating_places_and_restaurants, candy_nut_and_Confectionery_stores, drug_stores_and_pharmacies,
    merchandise_services_and_debt_repayment, mastercard_moneysend, fast_food_restaurants, cosmetic_stores, family_clothing_stores,
    lumber_building_materials_stores, variety_stores, hotels_motels_and_resorts, men_s_women_s_clothing_stores, drinking_places,
    shoe_stores, confectionery, miscellaneous_business_services, motion_picture_theaters_cinema, betting_casino_gambling,
    department_stores, direct_marketing_merchant, children_s_and_infants_wear_stores, non_financial_institutions,
    household_appliance_stores, direct_marketing_catalog_merchant, sporting_goods_stores, duty_free_stores, hobby_toy_and_game_shops,
    sports_and_riding_apparel_stores, beauty_and_barber_shops, women_s_accessory_and_specialty_shops, package_stores_beer_wine_and_liquor,
    women_s_ready_to_wear_stores, professional_services, miscellaneous_and_specialty_retail_shops, book_stores, advertising_services,
    video_game_arcades_establishments, electronics_stores, chemicals_and_allied_products, membership_clubs_sports_recreation_athletic,
    gift_card_novelty_and_souvenir_shops, jewelry_stores_watches_clocks_and_silverware_stores, medical_services_and_health_practitioners,
    miscellaneous_apparel_and_accessory_shops, ticket_agencies_and_theatrical_producers, pet_shops_pet_foods_and_supplies_stores,
    furniture_home_furnishings_and_equipment_stores, men_s_and_boys_clothing_and_accessories_stores, miscellaneous_general_merchandise,
    health_and_beauty_spas, home_supply_warehouse_stores, record_stores, recreation_services, parking_lots_parking_meters_and_garages,
    used_merchandise_and_secondhand_stores, government_services, US_federal_government_agencies_or_departments, computer_software_stores,
    stationery_stores_office_and_school_supply_stores, florists_supplies_nursery_stock_and_flowers, freezer_and_locker_meat_provisioners,
    laundry_cleaning_and_garment_services, florists, opticians_optical_goods_and_eyeglasses, automotive_parts_and_accessories_stores,
    medical_and_dental_laboratories, car_rental_agencies, cigar_stores_and_stands, commercial_sports_professional_sports_clubs,
    tourist_attractions_and_exhibits, dairy_products_stores, miscellaneous_personal_services, glassware_crystal_stores,
    miscellaneous_home_furnishing_specialty_stores, luggage_and_leather_goods_stores, sewing_needlework_fabric_and_piece_goods_stores,
    information_retrieval_services, motion_picture_and_video_tape_production_and_distribution, dentists_and_orthodontists,
    computers_and_computer_peripheral_equipment_and_software, fuel_dealers_fuel_oil_wood_coal_and_liquefied_petroleum,
    bakeries_0, charitable_social_service_organizations)
    return mcc_coding_list


country_dict = {'GE': ['GE', 'GEGE'],
 'MY': ['MY', 'MYMY'],
 'AU': ['AU', 'AUAU'],
 'CN': ['CN', 'CNCN'],
 'KZ': ['KZ', 'KZKZ'],
 'MK': ['MK', 'MKMK'],
 'CH': ['CH', 'CHCH'],
 'EG': ['EG', 'EGEG'],
 'NL': ['NL', 'NLNL'],
 'DO': ['DO', 'DODO'],
 'IT': ['IT', 'ITIT'],
 'DE': ['DE', 'DEDE'],
 'SE': ['SE', 'SESE'],
 'TR': ['TR', 'TRTR'],
 'GR': ['GR', 'GRGR'],
 'ME': ['ME', 'MEME'],
 'SI': ['SI', 'SISI'],
 'IN': ['IN', 'ININ'],
 'NO': ['NO', 'NONO'],
 'RU': ['RU', 'RURU'],
 'BY': ['BY', 'BYBY'],
 'IE': ['IE', 'IEIE'],
 'RO': ['RO', 'RORO'],
 'CY': ['CY', 'CYCY'],
 'SK': ['SK', 'SKSK'],
 'DK': ['DK', 'DKDK'],
 'HK': ['HK', 'HKHK'],
 'HU': ['HU', 'HUHU'],
 'AE': ['AE', 'AEAE'],
 'RS': ['RS', 'RSRS'],
 'PL': ['PL', 'PLPL'],
 'ES': ['ES', 'ESES'],
 'LV': ['LV', 'LVLV'],
 'MC': ['MC', 'MCMC'],
 'GB': ['GB', 'GBGB'],
 'IL': ['IL', 'ILIL'],
 'CZ': ['CZ', 'CZCZ'],
 'MU': ['MU', 'MUMU'],
 'BE': ['BE', 'BEBE'],
 'AT': ['AT', 'ATAT'],
 'FR': ['FR', 'FRFR'],
 'PT': ['PT', 'PTPT'],
 'HR': ['HR', 'HRHR'],
 'AD': ['AD', 'ADAD'],
 'GI': ['GI', 'GIGI'],
 'BG': ['BG', 'BGBG'],
 'MD': ['MD', 'MDMD'],
 'LU': ['LU', 'LULU'],
 'UA': ['. UA', 'A UA', 'K UA', 'SKUA', 'UAUA', 'UA'],
 'US': ['ARUS',
  'AZUS',
  'CAUS',
  'DCUS',
  'DEUS',
  'FLUS',
  'GAUS',
  'IAUS',
  'IDUS',
  'ILUS',
  'KSUS',
  'MDUS',
  'MNUS',
  'MOUS',
  'NCUS',
  'NEUS',
  'NJUS',
  'NVUS',
  'NYUS',
  'OHUS',
  'ORUS',
  'PAUS',
  'TNUS',
  'TXUS',
  'USUS',
  'UTUS',
  'WAUS',
  'WIUS'],
 'MT': ['MT', 'MTMT'],
 'BR': ['BR', 'BRBR'],
 'FI': ['FI', 'FIFI'],
 'TH': ['TH', 'THTH'],
 'GH': ['GH', 'GHGH'],
 'ID': ['ID', 'IDID'],
 'CR': ['CR', 'CRCR'],
 'CA': ['CA', 'ONCA', 'CACA'],
 'SG': ['SG', 'SGSG'],
 'NDFCountry': ['',
  'oCHCH',
  'EE',
  'AM',
  'SN',
  'LT',
  'BF',
  'KW',
  '™rCHCH',
  'MGMG',
  'KN',
  'BB',
  'PEPE',
  'AG',
  'CW',
  'lHUHU',
  'JO',
  'MV',
  'AZ',
  'LK',
  'PYPY',
  'JPJP',
  'VN',
  'TJ',
  'ML',
  'TG',
  'vHUHU',
  'PR']}


def get_train_data(FILE_NAME, FILE_PATH):
    temp = pd.read_csv(FILE_PATH + FILE_NAME)
    return temp


def get_categorical_data():
    dtrain = get_train_data(FILE_NAME, FILE_PATH)
    dtest = get_train_data(TEST_FILE_NAME, FILE_PATH)
    



#    dtrain_cat = dtrain.drop(['cgsettlementbufferid', 'clientid', 'sexid', 'amount'], axis=1, inplace=False)
#    dtest_cat = dtest.drop(['cgsettlementbufferid', 'clientid', 'amount'], axis=1, inplace=False )
#    train_index = dtrain_cat.index.tolist()

#    df_cat = dtrain_cat.append(dtest_cat, ignore_index=True)
#    df_cat = dtrain.append(dtest, ignore_index=True)

#    del dtrain
#    del dtest
#    del dtrain_cat
#    del dtest_cat
    return dtrain, dtest


def get_day(timestamp):
    day = datetime.datetime.fromtimestamp(timestamp).strftime("%A")
    return day


def get_month(timestamp):
    month = datetime.datetime.fromtimestamp(timestamp).strftime("%b")
    return month


def get_year(timestamp):
    year = datetime.datetime.fromtimestamp(timestamp).strftime("%Y")
    return year


def get_hour(timestamp):
    hour = datetime.datetime.fromtimestamp(timestamp).strftime('%H')
    return hour


def get_country(local):
    country = local[36:].strip()
    return country 


def get_city(local):
    city = local[23:36].strip()
    return city


def get_inst(local):
    inst = local[0:22].strip()
    return inst


def get_normal_country(country):
    for key in country_dict.keys():
        if country in country_dict[key]:
            return key


def get_cities_dict(FILE_PATH, CITIES_FILE_NAME):
    cities_dict = dict()
    with open(FILE_PATH + '/' + CITIES_FILE_NAME,'r') as data:
        reader = csv.reader(data)
        for line in reader:
            line[:] = [item for item in line if item != '']
            cities_dict[line[0]] = sorted(list(set(line)))
    return cities_dict


def get_normal_city(city):
    for key in cities_dict.keys():
        #print(key, city)
        for city_d in cities_dict[key]:
            #print(city, city_d)
            if city == city_d:              
                return key
                break
    else:
        return 'no city'


def get_mcc_coding_dict(mcc_coding_list):
    mcc_res_dict = dict()
    for element in mcc_coding_list:
        mcc_res_dict[mcc_coding_list.index(element)] = element
    return mcc_res_dict  


def get_normal_mcc(mcc):
    for key in mcc_res_dict.keys():
        #print(key, city)
        for mcc_d in mcc_res_dict[key]:
            #print(type(mcc), type(mcc_d))
            if mcc == mcc_d:              
                return key
                break
    else:
        return 'no mcc'


#def process_data(df):
#    res_list = list()
#    id_list = dict(df.clientid.value_counts())
#    for client_id in id_list:
#        temp = df.loc[df['clientid'] == client_id]
#        res_list.append(temp)
#    return res_list


def process_data(df):
    res_list = list()
    id_list = dict(df.clientid.value_counts())
    for client_id in id_list:
        temp = df.loc[df['clientid'] == client_id]
        year_list = dict(df.year.value_counts())
        for year in year_list:
            temp_year = temp.loc[temp['year'] == year]
            if not temp_year.empty:
                month_list = dict(temp_year.month.value_counts())
                for month in month_list:
                    temp_month = temp_year.loc[temp_year['month'] == month]
                    if not temp_month.empty:
                        res_list.append(temp_month)
    return res_list


def frame_template(hour_list, 
                   day_list, 
                   month_list, 
                   cities_list, 
                   countries_list, 
                   mcc_list, 
                   average_month, 
                   average_day,
                   average_hour,
                   average_country,
                   average_mcc,
                   std_mcc,
                   std_day,
                   std_month,
                   std_hour,
                   std_country,
                   mean_day,
                   mean_month,
                   mean_mcc,
                   mean_hour,
                   mean_country,
                   var_day,
                   var_month,
                   var_mcc,
                   var_hour,
                   var_country,
                   max_day,
                   max_month,
                   max_mcc,
                   max_hour,
                   max_country,
                   size):
    res_df = pd.DataFrame(index=range(size), columns = (
            hour_list
            + 
            day_list
            + 
            month_list
            + 
            cities_list
            +
            countries_list
            +
            mcc_list
            +
            average_month
            +
            average_day
            +
            average_hour
            +
            average_country
            +
            average_mcc
            +
            std_mcc
            +
            std_day
            +
            std_month
            +
            std_hour
            +
            std_country
            +
            mean_day
            +
            mean_month
            +
            mean_mcc
            +
            mean_hour
            +
            mean_country
            +
            var_day
            +
            var_month
            +
            var_mcc
            +
            var_hour
            +
            var_country
            +
            max_day
            +
            max_month
            +
            max_mcc
            +
            max_hour
            +
            max_country
            +
            ['sexid']))    
    return res_df


######
######
    
def random_forest_tuning(X_train, y_train):
    
    param_grid = [
        {'n_estimators':[640, 660, 680, 700, 1000],
        'max_depth':[1, 2, 3, 4, 5, 10, 20],
        'bootstrap':[True, False],
        'max_features':['auto', 'sqrt'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }
    ]
    
    forest_class = ensemble.RandomForestClassifier()
    
    grid_search = GridSearchCV(forest_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    forest_cls = grid_search.best_estimator_
    
    print('RandomForest best params', '\n', grid_search.best_params_)
    print('RandomForest best score', '\n', grid_search.best_score_)
    
#    forest_cv_res = grid_search.cv_results_
    
#    for mean_score, params in zip(forest_cv_res['mean_test_score'], forest_cv_res['params']):
#        print(mean_score, params)
    
    return forest_cls


def linear_SVC_tuning(X_train, y_train):

    param_grid = [
        {'class_weight':[None, 'balanced'],
        'fit_intercept':[False, True],
        'dual':[True, False],
        'max_iter': [1000, 2000, 5000, 10000],
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'loss':['squared_hinge']
        }
    ]
    
    lin_class = LinearSVC()
    
    grid_search = GridSearchCV(lin_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    lin_cls = grid_search.best_estimator_
    
    print('linearSVC best params', '\n', grid_search.best_params_)
    print('linear_SVC best score', '\n', grid_search.best_score_)
    
    return lin_cls



###
###
def draft_classifiers_evaluation(df_res, y):

    classifiers = [
    LinearSVC(max_iter=2000),
    LinearSVR(C=0.01, max_iter=2000, dual=False, loss='squared_epsilon_insensitive'),
#    KNeighborsClassifier(3),
    SVC(probability=True),
    NuSVC(),
    DecisionTreeClassifier(),
    ensemble.RandomForestClassifier(n_estimators=680),
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
#    BernoulliNB(),
    LinearDiscriminantAnalysis(),
#    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
#    MLPClassifier(hidden_layer_sizes=(110, ), max_iter=800),
    SGDClassifier(loss='log', max_iter=800),
    LogisticRegressionCV(max_iter=800)]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(df_res, y):
        X_train, X_test = df_res.iloc[train_index], df_res.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
            acc = roc_auc_score(y_test, train_predictions)


            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

#    print(acc_dict)
    print(log)
    pass


def classifiers_evaluation(df_res, y):

    classifiers = [
    LinearSVC(max_iter=2000),
    LinearSVR(C=0.01, max_iter=2000, dual=False, loss='squared_epsilon_insensitive'),
#    KNeighborsClassifier(3),
    SVC(probability=True),
    NuSVC(),
#    DecisionTreeClassifier(),
    ensemble.RandomForestClassifier(n_estimators=680),
    ensemble.AdaBoostClassifier(),
    ensemble.GradientBoostingClassifier(),
#    BernoulliNB(),
    LinearDiscriminantAnalysis(),
#    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
#    MLPClassifier(hidden_layer_sizes=(110, ), max_iter=800),
    SGDClassifier(loss='log', max_iter=800),
    LogisticRegressionCV(max_iter=800)]

    log_cols = ["Classifier", "ROC_AUC score"]
    log = pd.DataFrame(columns=log_cols)

    quantile = preprocessing.QuantileTransformer(n_quantiles=2500)
    X = quantile.fit_transform(df_res)
    
#    minmax = preprocessing.MinMaxScaler()
#    X = minmax.fit_transform(df_res)
    
#    norm = preprocessing.Normalizer()
#    X = norm.fit_transform(df_res) 

#    standart = preprocessing.StandardScaler()
#    X = standart.fit_transform(df_res)
    
#    robust = preprocessing.RobustScaler()
#    X = robust.fit_transform(df_res)  
    
#    maxabs = preprocessing.MaxAbsScaler()
#    X = maxabs.fit_transform(df_res)    

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for clf in classifiers:
            name = clf.__class__.__name__
            clf.fit(X_train, y_train)
            train_predictions = clf.predict(X_test)
#            acc = accuracy_score(y_test, train_predictions)
            acc = roc_auc_score(y_test, train_predictions)


            if name in acc_dict:
                acc_dict[name] += acc
            else:
                acc_dict[name] = acc

    for clf in acc_dict:
        acc_dict[clf] = acc_dict[clf] / 10.0
        log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
        log = log.append(log_entry)

#    print(acc_dict)
    print(log)


if __name__ == '__main__':
    df_train, df_test = get_categorical_data()
    
    df_train['year'] = df_train['trandatetime'].apply(lambda x: get_year(x))
    df_train['month'] = df_train['trandatetime'].apply(lambda x: get_month(x))
    df_train['day'] = df_train['trandatetime'].apply(lambda x: get_day(x))
    df_train['hour'] = df_train['trandatetime'].apply(lambda x: get_hour(x))
    df_train['country'] = df_train['location'].apply(lambda x: get_country(x))
    df_train['city'] = df_train['location'].apply(lambda x: get_city(x))
    df_train['institution'] = df_train['location'].apply(lambda x: get_inst(x))

    df_test['year'] = df_test['trandatetime'].apply(lambda x: get_year(x))
    df_test['month'] = df_test['trandatetime'].apply(lambda x: get_month(x))
    df_test['day'] = df_test['trandatetime'].apply(lambda x: get_day(x))
    df_test['hour'] = df_test['trandatetime'].apply(lambda x: get_hour(x))
    df_test['country'] = df_test['location'].apply(lambda x: get_country(x))
    df_test['city'] = df_test['location'].apply(lambda x: get_city(x))
    df_test['institution'] = df_test['location'].apply(lambda x: get_inst(x))

    cities_dict = get_cities_dict(FILE_PATH, CITIES_FILE_NAME)

    df_train['city'] = df_train['city'].apply(lambda x: get_normal_city(x))
    df_test['city'] = df_test['city'].apply(lambda x: get_normal_city(x))    
    
    df_train['country'] = df_train['country'].apply(lambda x: get_normal_country(x))
    df_test['country'] = df_test['country'].apply(lambda x: get_normal_country(x))
            
    mcc_coding_list = get_mcc_coding_list()

    mcc_res_dict = get_mcc_coding_dict(mcc_coding_list)
    
    mcc_res_dict = {f'{k}_mcc': v for k, v in mcc_res_dict.items()}
    
    df_train['mcc'] = df_train['mcc'].apply(lambda x: get_normal_mcc(x))
    df_test['mcc'] = df_test['mcc'].apply(lambda x: get_normal_mcc(x))
    
    hour_list = list(df_train.hour.unique())
    day_list = list(df_train.day.unique())
    month_list = list(df_train.month.unique())
    cities_list = list(set(list(df_train.city.unique()) + list(df_test.city.unique())))
    countries_list = list(set(list(df_train.country.unique()) + list(df_test.country.unique())))
    mcc_list = list(set(list(df_train.mcc.unique()) + list(df_test.mcc.unique())))
    average_month = [f'{k}_avg' for k in month_list]
    average_day = [f'{k}_avg' for k in day_list]
    average_hour = [f'{k}_avg' for k in hour_list]
    average_country = [f'{k}_avg' for k in countries_list]
    average_mcc = [f'{k}_avg' for k in mcc_list]
    std_mcc = [f'{k}_std' for k in mcc_list]
    std_day = [f'{k}_std' for k in day_list]
    std_month = [f'{k}_std' for k in month_list]
    std_hour = [f'{k}_std' for k in hour_list]
    std_country = [f'{k}_std' for k in countries_list]
    mean_day = [f'{k}_mean' for k in day_list]
    mean_month = [f'{k}_mean' for k in month_list]
    mean_mcc = [f'{k}_mean' for k in mcc_list]
    mean_hour = [f'{k}_mean' for k in hour_list]
    mean_country = [f'{k}_mean' for k in countries_list]
    var_day = [f'{k}_var' for k in day_list]
    var_month = [f'{k}_var' for k in month_list]
    var_mcc = [f'{k}_var' for k in mcc_list]
    var_hour = [f'{k}_var' for k in hour_list]
    var_country = [f'{k}_var' for k in countries_list]
    max_day = [f'{k}_max' for k in day_list]
    max_month = [f'{k}_max' for k in month_list]
    max_mcc = [f'{k}_max' for k in mcc_list]
    max_hour = [f'{k}_max' for k in hour_list]
    max_country = [f'{k}_max' for k in countries_list]
        
    train_list = process_data(df_train)
    test_list = process_data(df_test)
    
    train_data = frame_template(
            hour_list,
            day_list,
            month_list,
            cities_list,
            countries_list,
            mcc_list,
            average_month,
            average_day,
            average_hour,
            average_country,
            average_mcc,
            std_mcc,
            std_day,
            std_month,
            std_hour,
            std_country,
            mean_day,
            mean_month,
            mean_mcc,
            mean_hour,
            mean_country,
            var_day,
            var_month,
            var_mcc,
            var_hour,
            var_country,
            max_day,
            max_month,
            max_mcc,
            max_hour,
            max_country,
            len(train_list)
            )
    
    for I in range(0,len(train_list)):
        
        for key in list(dict(train_list[I].city.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].city.value_counts())[key]
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].country.value_counts())[key]
            
        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].day.value_counts())[key]
            
        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].hour.value_counts())[key]
            
        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].month.value_counts())[key]
            
        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key][I] = dict(train_list[I].mcc.value_counts())[key]
#mean
        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key + '_avg'][I] = train_list[I][train_list[I]['month'] == key]['amount'].mean() // 100

        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key + '_avg'][I] = train_list[I][train_list[I]['day'] == key]['amount'].mean() // 100

        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key + '_avg'][I] = train_list[I][train_list[I]['hour'] == key]['amount'].mean() // 100
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key + '_avg'][I] = train_list[I][train_list[I]['country'] == key]['amount'].mean() // 100

        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key + '_avg'][I] = train_list[I][train_list[I]['mcc'] == key]['amount'].mean() // 100
#std
        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key + '_std'][I] = train_list[I][train_list[I]['mcc'] == key]['amount'].std() // 100
            
        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key + '_std'][I] = train_list[I][train_list[I]['day'] == key]['amount'].std() // 100

        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key + '_std'][I] = train_list[I][train_list[I]['month'] == key]['amount'].std() // 100
            
        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key + '_std'][I] = train_list[I][train_list[I]['hour'] == key]['amount'].std() // 100
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key + '_std'][I] = train_list[I][train_list[I]['country'] == key]['amount'].std() // 100
#median
        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key + '_mean'][I] = train_list[I][train_list[I]['day'] == key]['amount'].median() // 100
            
        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key + '_mean'][I] = train_list[I][train_list[I]['month'] == key]['amount'].median() // 100
            
        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key + '_mean'][I] = train_list[I][train_list[I]['mcc'] == key]['amount'].median() // 100
            
        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key + '_mean'][I] = train_list[I][train_list[I]['hour'] == key]['amount'].median() // 100
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key + '_mean'][I] = train_list[I][train_list[I]['country'] == key]['amount'].median() // 100        
#variance
        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key + '_var'][I] = train_list[I][train_list[I]['day'] == key]['amount'].var() // 100
            
        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key + '_var'][I] = train_list[I][train_list[I]['month'] == key]['amount'].var() // 100
            
        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key + '_var'][I] = train_list[I][train_list[I]['mcc'] == key]['amount'].var() // 100
            
        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key + '_var'][I] = train_list[I][train_list[I]['hour'] == key]['amount'].var() // 100
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key + '_var'][I] = train_list[I][train_list[I]['country'] == key]['amount'].var() // 100 
#max
        for key in list(dict(train_list[I].day.value_counts()).keys()):
            train_data[key + '_max'][I] = train_list[I][train_list[I]['day'] == key]['amount'].max() // 100
            
        for key in list(dict(train_list[I].month.value_counts()).keys()):
            train_data[key + '_max'][I] = train_list[I][train_list[I]['month'] == key]['amount'].max() // 100
            
        for key in list(dict(train_list[I].mcc.value_counts()).keys()):
            train_data[key + '_max'][I] = train_list[I][train_list[I]['mcc'] == key]['amount'].max() // 100
            
        for key in list(dict(train_list[I].hour.value_counts()).keys()):
            train_data[key + '_max'][I] = train_list[I][train_list[I]['hour'] == key]['amount'].max() // 100
            
        for key in list(dict(train_list[I].country.value_counts()).keys()):
            train_data[key + '_max'][I] = train_list[I][train_list[I]['country'] == key]['amount'].max() // 100 
            
        train_data['sexid'][I] = train_list[I].sexid.unique()[0]
        
    train_data.fillna(0, inplace=True)
    
    y_train = train_data['sexid']
    X_train = train_data.drop('sexid', axis=1)


    rand_for = random_forest_tuning(X_train, y_train)
    lin_svc = linear_SVC_tuning(X_train, y_train)
    
    X_train_full = X_train.apply(np.log)
    X_train_full[np.isneginf(X_train_full)] = 0
    
    rand_for = random_forest_tuning(X_train_full, y_train)
    lin_svc = linear_SVC_tuning(X_train_full, y_train)
    
#    from keras import losses
#    from keras import metrics
#    from keras import models
#    from keras import layers
    from keras import regularizers
#    from keras import optimizers
    from sklearn.preprocessing import MinMaxScaler


    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import SGD
    
    model = Sequential()
    model.add(Dense(128, input_dim=1492, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.009)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    sgd = SGD(lr=0.00015, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
#                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
#    model = models.Sequential()
##    model.add(layers.Dense(1492, activation='relu', kernel_regularizer=regularizers.l2(0.25), input_shape=(1492,)))
#    model.add(layers.Dense(8, activation='relu'))
#    model.add(layers.Dense(6, activation='relu'))
#    model.add(layers.Dense(4, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
##    model.add(layers.Dense(50, activation='relu'))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    model.compile(optimizer='rmsprop',
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
    
#    x_val = X_train[:1000].values
#    partial_x_train = X_train[1000:].values
#    
#    y_val = y_train[:1000].values
#    partial_y_train = y_train[1000:].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_min = scaler.fit_transform(X_train_full)
    
    x_val = X_train_min[:1000]
    partial_x_train = X_train_min[1000:]
    
    y_val = y_train[:1000].values
    partial_y_train = y_train[1000:].values
    

#    x_val = X_train_full[:1000].values
#    partial_x_train = X_train_full[1000:].values
#    
#    y_val = y_train[:1000].values
#    partial_y_train = y_train[1000:].values
    
    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=150,
                    batch_size=32,
                    validation_data=(x_val, y_val))
    

#### оценка классификаторв с преобработкой данных и без
#######
#######
#    classifiers_evaluation(X_train, y_train)
#    print('\n', 'Clear evaluation without data preprocessing', '\n')
#    draft_classifiers_evaluation(X_train, y_train)




#    X_train, X_test, y_train, y_test  =  train_test_split(
#                                            X_train,
#                                            y_train,
#                                            test_size = 0.25,
#                                            random_state=42
#                                            )
#        
#    model_0 = RandomForestClassifier(n_estimators=700, n_jobs=-1)
#    model_0.fit(X_train, y_train)
#    y_pred_forest_short = model_0.predict(X_test)
#    print(classification_report(y_test, y_pred_forest_short))
#    print(roc_auc_score(y_test, y_pred_forest_short))    

#    param_grid = [
#        {'n_estimators':[640, 660, 680, 700, 1000],
#        'max_depth':[1, 2, 3, 4, 5, 10, 20],
#        'bootstrap':[True, False],
#        'max_features':['auto', 'sqrt'],
#        'min_samples_split': [2, 5, 10],
#        'min_samples_leaf': [1, 2, 4]
#        }
#    ]
#    
#    forest_class = ensemble.RandomForestClassifier()
#    
#    grid_search = GridSearchCV(forest_class, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
#    
#    grid_search.fit(X_train, y_train)
#    
#    forest_cls = grid_search.best_estimator_
#    
#    print('RandomForest best params', '\n', grid_search.best_params_)
#    
#    forest_cv_res = grid_search.cv_results_
#    
#    for mean_score, params in zip(forest_cv_res['mean_test_score'], forest_cv_res['params']):
#        print(mean_score, params)
#    X_train.drop(hour_list, axis=1, inplace=True)
#    X_train.drop(day_list, axis=1, inplace=True)
#    X_train.drop(month_list, axis=1, inplace=True)
#    X_train.drop(cities_list, axis=1, inplace=True)
#    X_train.drop(countries_list, axis=1, inplace=True)
#    X_train.drop(mcc_list, axis=1, inplace=True)