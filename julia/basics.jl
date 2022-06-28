using CSV
using DataFrames


visa_2020 = CSV.read("/Users/wastechs/Documents/data/financials/2020_visa_transactions.csv", DataFrame)
visa_2021 = CSV.read("/Users/wastechs/Documents/data/financials/2021_visa_transactions.csv", DataFrame)
visa_2022 = CSV.read("/Users/wastechs/Documents/data/financials/2022-06-05_transaction.csv", DataFrame)

subset(visa_2020(), :category => ByRow())