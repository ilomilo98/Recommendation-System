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
# ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

# we are going to choose germany in this case.
df_gr = df[df['Country'] == "Germany"]


df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20) # her bir faturada olan ürünler ve ürünlerin adedi

df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5] #unstack ile pivot şekline getiriyoruz. Ürün adları sütunlara fatura noları satırlara

df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5] #boş değerleri 0 ile dolduruyoruz.

##apply komutu satır yada sutunu dolaşıyor. Applymap tüm hücreleri. Bize ürün frekansı değil var mı yok mu bilgisi lazım o yuzden yoksa 0 varsa 1 yazdırcaz.
df_gr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

df_gr.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5] #hepsinin toplanmış hali

#### fatura ürün bilgisinin oluşturulması stock kodu da eklendi yukarıdakinin fonksiyon hali .. 1  0 şeklinde#####
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


gr_inv_pro_df = create_invoice_product_df(df_gr)
gr_inv_pro_df.head()

gr_inv_pro_df = create_invoice_product_df(df_gr, id=True)

### birlikteliklerin olasılıkları ##

#X ve Y birlikte görülme olasılığı ekmek süt
#Support(X,Y) = Freq(X,Y)/N
#Confidence(X,Y) = Freq(X,Y) /Freq(X) ekmek satın alındığında sütün alınması olasılığı
#Lift= Support (X,y)/(Support(X) * Support(Y) X satın alındığında Ynin satın alınma olasılığı lift kat kadar artar.
#Sepet tanımı önemlidir.
#Veri içerisindeki örüntüleri bulmak için kullanılan kural tabanlı bir makine öğrenmesi tekniğidir.. Apriori algoritması

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values.tolist()
    print(product_name)

# TASK 3-ID'leri verilen ürünlerin isimleri?
#Kullanıcı 1 ürün id'si: 21987 Kullanıcı 2 ürün id'si: 23235 Kullanıcı 3 ürün id'si: 22747
check_id(df_gr, 21987)
check_id(df_gr, 23235)
check_id(df_gr, 22747)

#TASK 4- Sepetteki kullanıcılar için ürün önerisi, yalnız şöyle bir ince çizgi var.
# Müşteri bu önerilen ürünü satın almış olabilir olmayadabilir.
# Şuan çok önemli değil ama real casede önemli olabilir..
frequent_itemsets = apriori(gr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)

# ürün önerileri #
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

#kullanıcı 1#
arl_recommender(rules, 21987,2) # sonuç : [21989, 21086]
check_id(df_gr, 21989) #['PACK OF 20 SKULL PAPER NAPKINS']
check_id(df_gr, 21086) #['SET/6 RED SPOTTY PAPER CUPS']

#kullanıcı 2#
arl_recommender(rules, 23235,3) #[23243, 23244, 23240]
check_id(df_gr, 23243) #['SET OF TEA COFFEE SUGAR TINS PANTRY']
check_id(df_gr, 23244) #['ROUND STORAGE TIN VINTAGE LEAF']
check_id(df_gr, 23240) #['SET OF 4 KNICK KNACK TINS DOILEY ']

#kullanıcı 3#
arl_recommender(rules, 22747,1) #[22746]
check_id(df_gr, 22746)  #["POPPY'S PLAYHOUSE LIVINGROOM "]
