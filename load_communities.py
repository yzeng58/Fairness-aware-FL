######################################################
### Pre-processing code (leave here for reference) ###
######################################################
# import pandas as pd
# import os

# df = pd.read_csv(os.path.join('communities', 'communities.data'), header = None, na_values = '?')
# df.columns = ["state"
# ,"county"
# ,"community"
# ,"communityname"
# ,"fold"
# ,"population"
# ,"householdsize"
# ,"racepctblack"
# ,"racePctWhite"
# ,"racePctAsian"
# ,"racePctHisp"
# ,"agePct12t21"
# ,"agePct12t29"
# ,"agePct16t24"
# ,"agePct65up"
# ,"numbUrban"
# ,"pctUrban"
# ,"medIncome"
# ,"pctWWage"
# ,"pctWFarmSelf"
# ,"pctWInvInc"
# ,"pctWSocSec"
# ,"pctWPubAsst"
# ,"pctWRetire"
# ,"medFamInc"
# ,"perCapInc"
# ,"whitePerCap"
# ,"blackPerCap"
# ,"indianPerCap"
# ,"AsianPerCap"
# ,"OtherPerCap"
# ,"HispPerCap"
# ,"NumUnderPov"
# ,"PctPopUnderPov"
# ,"PctLess9thGrade"
# ,"PctNotHSGrad"
# ,"PctBSorMore"
# ,"PctUnemployed"
# ,"PctEmploy"
# ,"PctEmplManu"
# ,"PctEmplProfServ"
# ,"PctOccupManu","PctOccupMgmtProf"
# ,"MalePctDivorce"
# ,"MalePctNevMarr"
# ,"FemalePctDiv"
# ,"TotalPctDiv"
# ,"PersPerFam"
# ,"PctFam2Par"
# ,"PctKids2Par"
# ,"PctYoungKids2Par"
# ,"PctTeen2Par"
# ,"PctWorkMomYoungKids"
# ,"PctWorkMom"
# ,"NumIlleg"
# ,"PctIlleg"
# ,"NumImmig"
# ,"PctImmigRecent"
# ,"PctImmigRec5"
# ,"PctImmigRec8"
# ,"PctImmigRec10"
# ,"PctRecentImmig"
# ,"PctRecImmig5"
# ,"PctRecImmig8"
# ,"PctRecImmig10"
# ,"PctSpeakEnglOnly"
# ,"PctNotSpeakEnglWell"
# ,"PctLargHouseFam"
# ,"PctLargHouseOccup"
# ,"PersPerOccupHous"
# ,"PersPerOwnOccHous"
# ,"PersPerRentOccHous"
# ,"PctPersOwnOccup"
# ,"PctPersDenseHous"
# ,"PctHousLess3BR"
# ,"MedNumBR"
# ,"HousVacant"
# ,"PctHousOccup"
# ,"PctHousOwnOcc"
# ,"PctVacantBoarded"
# ,"PctVacMore6Mos"
# ,"MedYrHousBuilt"
# ,"PctHousNoPhone"
# ,"PctWOFullPlumb"
# ,"OwnOccLowQuart"
# ,"OwnOccMedVal"
# ,"OwnOccHiQuart"
# ,"RentLowQ"
# ,"RentMedian"
# ,"RentHighQ"
# ,"MedRent"
# ,"MedRentPctHousInc"
# ,"MedOwnCostPctInc"
# ,"MedOwnCostPctIncNoMtg"
# ,"NumInShelters"
# ,"NumStreet"
# ,"PctForeignBorn"
# ,"PctBornSameState"
# ,"PctSameHouse85"
# ,"PctSameCity85"
# ,"PctSameState85"
# ,"LemasSwornFT"
# ,"LemasSwFTPerPop"
# ,"LemasSwFTFieldOps"
# ,"LemasSwFTFieldPerPop"
# ,"LemasTotalReq"
# ,"LemasTotReqPerPop"
# ,"PolicReqPerOffic"
# ,"PolicPerPop"
# ,"RacialMatchCommPol"
# ,"PctPolicWhite"
# ,"PctPolicBlack"
# ,"PctPolicHisp"
# ,"PctPolicAsian"
# ,"PctPolicMinor"
# ,"OfficAssgnDrugUnits"
# ,"NumKindsDrugsSeiz"
# ,"PolicAveOTWorked"
# ,"LandArea"
# ,"PopDens"
# ,"PctUsePubTrans"
# ,"PolicCars"
# ,"PolicOperBudg"
# ,"LemasPctPolicOnPatr"
# ,"LemasGangUnitDeploy","LemasPctOfficDrugUn"
# ,"PolicBudgPerPop"
# ,"ViolentCrimesPerPop"]
# for column in ['racePctWhite', 'racepctblack', 'racePctAsian', 'racePctHisp']:
#     qt = df[column].quantile(.5)
#     high_idx = np.where(df[column] > qt)[0]
#     low_idx = np.where(df[column] <= qt)[0]
#     df.loc[high_idx, column] = 1
#     df.loc[low_idx, column] = 0

# high_idx = np.where(df['ViolentCrimesPerPop'] > .7)[0]
# low_idx = np.where(df['ViolentCrimesPerPop'] <= .7)[0]
# df.loc[high_idx, 'ViolentCrimesPerPop'] = 1
# df.loc[low_idx, 'ViolentCrimesPerPop'] = 0
# df = df.dropna(axis = 1)
# df.to_csv(os.path.join('communities', 'communities_process.csv'))
######################################################

import numpy as np
from utils import *
import torch, os
import pandas as pd

if __name__ == "load_communities":
    np.random.seed(1)
    torch.manual_seed(0)
    
    sensitive_attributes = ['racePctWhite', 'racepctblack', 'racePctAsian', 'racePctHisp']
    categorical_attributes = []
    df = pd.read_csv(os.path.join('..', 'communities', 'communities_process.csv'))
    features_to_keep = list(set(df.columns) - {'communityname'})
    continuous_attributes = list(set(features_to_keep) - {'racePctWhite', 'racepctblack', 'racePctAsian', 'racePctHisp', 'state'})
    label_name = 'ViolentCrimesPerPop'

    communities = process_csv('communities', 'communities_process.csv', label_name, 1, sensitive_attributes, None, categorical_attributes, continuous_attributes, features_to_keep)
    communities = communities.sample(frac=1).reset_index(drop=True)
    train = communities.iloc[:int(len(communities)*.9)]
    test = communities.iloc[int(len(communities)*.9):]

    state_high_idx = np.where(train.state > 20)[0]
    state_low_idx = np.where(train.state <= 20)[0]
    client1_idx = train[train.state > 20].index
    client2_idx = train[train.state <= 20].index
    train = train.drop(columns = ['state'])
    test = test.drop(columns = ['state'])
    communities_mean_sensitive = train['z'].mean()
    communities_z = len(set(communities.z))

    clients_idx = [client1_idx, client2_idx]

    communities_num_features = len(train.columns) - 1
    communities_train = LoadData(train, label_name, 'z')
    communities_test = LoadData(test, label_name, 'z')
    
    communities_info = [communities_train, communities_test, clients_idx]


    

