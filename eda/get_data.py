#!/usr/bin/env python3
import pandas as pd
from datetime import date
from sys import argv
from os.path import expanduser, isdir, join
from os import mkdir
from yaml import load
from nass import NassApi


def query_acres_planted(crop, min_year, max_year):
    crop_dict = {'corn': 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
                 'soybean': 'SOYBEANS - YIELD, MEASURED IN BU / ACRE'}
    if crop not in crop_dict.keys():
        raise ValueError('{} not a valid crop. Valid crop names are: {}'.format(crop, crop_dict.keys()))
    nass_api = NassApi(nass_key)
    nass_query = nass_api.query()

    years = [str(y) for y in range(min_year, max_year+1)]
    nass = nass_query.filter('source_desc', ['SURVEY'])\
                     .filter('short_desc', crop_dict[crop])\
                     .filter('year', years)\
                     .filter('sector_desc', 'CROPS')\
                     .filter('agg_level_desc', 'COUNTY')

    return nass.execute()

def get_nass_yields(crop, save_dir, min_year, max_year=None):

    if not max_year:
        max_year = date.today().year

    queried_data = query_acres_planted(crop, min_year, max_year)
    df = pd.DataFrame(queried_data)

    df['fips_code'] = df['state_fips_code'] + df['county_code']
    cols_to_keep = ['fips_code', 'state_alpha', 'year', 'Value', 'county_name', 'asd_desc', 'location_desc']
    df = df[cols_to_keep]
    df.rename(columns={'Value': 'nassyield'}, inplace=True)
    df['nassyield'] = df.nassyield.astype(float)
    # remove all invalid fips_codes (ending with '998')
    clean_df = df[df.fips_code.map(lambda i: not i.endswith('998'))]

    file_name = '{}_historical_yields_{}_{}.csv'.format(crop, min_year, max_year)
    clean_df.to_csv(join(save_dir + file_name))
    return clean_df

# arguments
credentials = load(open(expanduser('~/.scripts/credentials.yml')))
nass_key = credentials['nass_api_key']
target_crop = str(argv[1]) #'corn'
save_dir = '/Users/adamszabunio/Desktop/NASS_raw_data/{}/'.format(target_crop)
if not isdir(save_dir):
    mkdir(save_dir)
min_yield_year = int(argv[2]) #1970
max_yield_year = int(argv[3]) #1974

# get nass yield data
get_nass_yields(target_crop, save_dir, min_yield_year, max_yield_year)
