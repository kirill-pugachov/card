# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:27:06 2018

@author: User
"""


import pandas as pd
import time
import datetime
import csv

FILE_NAME = 'train_ds.csv'
FILE_PATH = 'Data/'
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
    
    df_train['mcc'] = df_train['mcc'].apply(lambda x: get_normal_mcc(x))
    df_test['mcc'] = df_test['mcc'].apply(lambda x: get_normal_mcc(x))
    
#    df_cat['week_day'] = df_cat['trandatetime'].apply(lambda x: get_day(x))
#    df_cat['month'] = df_cat['trandatetime'].apply(lambda x: get_month(x))
#    df_cat['year'] = df_cat['trandatetime'].apply(lambda x: get_year(x))





#    dtrain = get_train_data(FILE_NAME, FILE_PATH)
#    dtest = get_train_data(TEST_FILE_NAME, FILE_PATH)
#
#    dtrain_cat = dtrain.drop(['cgsettlementbufferid', 'clientid', 'sexid', 'amount'], axis=1, inplace=False)
#    dtest_cat = dtest.drop(['cgsettlementbufferid', 'clientid', 'amount'], axis=1, inplace=False )
#    train_index = dtrain_cat.index.tolist()
#    df_cat = dtrain_cat.append(dtest_cat, ignore_index=True)
#    del dtrain
#    del dtest
#    del dtrain_cat
#    del dtest_cat
