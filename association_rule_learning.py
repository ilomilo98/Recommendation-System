df_ = pd.read_excel(".../online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

dataframe.dropna(inplace=True)
dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
dataframe = dataframe[dataframe["Quantity"] > 0]
dataframe = dataframe[dataframe["Price"] > 0]
replace_with_thresholds(dataframe, "Quantity")
replace_with_thresholds(dataframe, "Price")    
    
df = retail_data_prep(df)

df.head() # take a look
    
############################################
# Preparing ARL Data Structure (Invoice-Product Matrix)
############################################

# we are going to choose germany in this case.
df_gr = df[df['Country'] == "Germany"]


df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20) # products and the number of products on each invoice

df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5] #We turn it into a pivot shape with unstack. Product names in columns, invoice numbers in rows

df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5] #We fill in blanks with 0.

##The apply command is looping through the row or column. Applymap all cells. We don't need the product frequency or not, so we will print 1 if there is 0.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

df_gr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


gr_inv_pro_df = create_invoice_product_df(df_gr)
gr_inv_pro_df.head()

gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)

### probabilities of associations ##

#X and Y probability of coexistence bread milk
#Support(X,Y) = Freq(X,Y)/N
#Confidence(X,Y) = probability of purchasing milk when Freq(X,Y) /Freq(X) bread is purchased
#Lift= Support (X,y)/(Support(X) * Support(Y) When X is purchased, the probability of Y being purchased increases by a factor of lift.
# Basket definition is important.
#It is a rule-based machine learning technique used to find patterns in data. Apriori algorithm

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values.tolist()
    print(product_name)

# The names of the products whose IDs are given?

check_id(df_gr, 21987)
check_id(df_gr, 23235)
check_id(df_gr, 22747)

#Product recommendation for users in the cart, but there is a fine line. The customer may or may not have purchased this recommended product.
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)

# product recommendations #
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

#Users 1#
arl_recommender(rules, 21987,2) # result : [21989, 21086]
check_id(df_gr, 21989) #['PACK OF 20 SKULL PAPER NAPKINS']
check_id(df_gr, 21086) #['SET/6 RED SPOTTY PAPER CUPS']

#User 2#
arl_recommender(rules, 23235,3) #[23243, 23244, 23240]
check_id(df_gr, 23243) #['SET OF TEA COFFEE SUGAR TINS PANTRY']
check_id(df_gr, 23244) #['ROUND STORAGE TIN VINTAGE LEAF']
check_id(df_gr, 23240) #['SET OF 4 KNICK KNACK TINS DOILEY ']

#User 3#
arl_recommender(rules, 22747,1) #[22746]
check_id(df_gr, 22746)  #["POPPY'S PLAYHOUSE LIVINGROOM "]
