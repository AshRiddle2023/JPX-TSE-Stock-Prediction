# load stock price data
df_price_raw = pd.read_csv(f"{train_files_dir}/stock_prices.csv")
price_cols = [
    "Date",
    "SecuritiesCode",
    "Close",
    "AdjustmentFactor",
    "ExpectedDividend"
]
df_price_raw = df_price_raw[price_cols]

# forecasting phase leaderboard:
df_price_supplemental = pd.read_csv(f"{supplemental_files_dir}/stock_prices.csv")
df_price_supplemental = df_price_supplemental[price_cols]
df_price_raw = pd.concat([df_price_raw, df_price_supplemental])

# filter data to reduce culculation cost 
df_price_raw = df_price_raw.loc[df_price_raw["Date"] >= "2022-07-01"]

# load Time Series API
import jpx_tokyo_market_prediction
# make Time Series API environment (this function can be called only once in a session)
env = jpx_tokyo_market_prediction.make_env()
# get iterator to fetch data day by day
iter_test = env.iter_test()

counter = 0
# fetch data day by day
for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    current_date = prices["Date"].iloc[0]
    sample_prediction_date = sample_prediction["Date"].iloc[0]
    print(f"current_date: {current_date}, sample_prediction_date: {sample_prediction_date}")

    if counter == 0:
        # to avoid data leakage
        df_price_raw = df_price_raw.loc[df_price_raw["Date"] < current_date]#current_date以前のデータにする

    # to generate AdjustedClose, increment price data
    df_price_raw = pd.concat([df_price_raw, prices[price_cols]])
    # generate AdjustedClose
    df_price = adjust_price(df_price_raw)

    # get target SecuritiesCodes
    codes = sorted(prices["SecuritiesCode"].unique())

    # generate feature
    feature = pd.concat([get_features_for_predict(df_price, code) for code in codes])
    # filter feature for this iteration
    feature = feature.loc[feature.index == current_date]

    # prediction
    feature.loc[:, "predict"] = feature["return_1day"] + feature["ExpectedDividend"]*100

    # set rank by predict
    feature = feature.sort_values("predict", ascending=True).drop_duplicates(subset=['SecuritiesCode'])
    feature.loc[:, "Rank"] = np.arange(len(feature))
    feature_map = feature.set_index('SecuritiesCode')['Rank'].to_dict()
    sample_prediction['Rank'] = sample_prediction['SecuritiesCode'].map(feature_map)

    # check Rank
    assert sample_prediction["Rank"].notna().all()
    assert sample_prediction["Rank"].min() == 0
    assert sample_prediction["Rank"].max() == len(sample_prediction["Rank"]) - 1

    # register your predictions
    env.predict(sample_prediction)
    counter += 1

! head submission.csv

! tail submission.csv
