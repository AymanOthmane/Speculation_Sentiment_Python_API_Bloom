import streamlit as st
import numpy as np
import time
from ssi_building import * 
from data_collection import *
from analysis import *
from Strategies_test import *

st.title("Speculation Sentiment")
onglet = st.sidebar.radio("Navigation", ["Introduction","Construction","Analyses","Stratégie"])

if onglet == "Introduction":
    st.markdown(
        """
        <style>
        .subheader-text {
            color: orange;
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='subheader-text'>Introduction de l'article</h1>", unsafe_allow_html=True)
    import streamlit as st

    st.markdown("""

### Auteur
Shaun William Davies

### Source
Journal of Financial and Quantitative Analysis, Vol. 57, No. 7, Novembre 2022

                
### Résumé
Les traders spéculatifs utilisent souvent dans leurs prises de décision des informations qu'ils considèrent comme privilégiées, même si elles sont souvent basées sur des signaux sans valeur. Cet article s'interesse à la demande des traders non informés et définit le sentiment spéculatif comme une croyance semblable à celle d'un joueur sur la direction future du marché. Davies se sert des ETF à effet de levieres ETF à effet de levier offrant une exposition amplifiée à des indices de marché pour créer un indice permettant de mesurer les activités d'arbitrage. Ces activités corrigent les erreurs de tarification entre les actions des ETF et leurs actifs sous-jacents et permettent de quantifier cette demande spéculative et son impact sur les prix du marché.

                
### Méthodologie
Pour construire le SSI, Davies utilise six des huit ETF à effet de levier d'origine : trois ETF longs à effet de levier (QLD, SSO, DDM) et trois ETF courts à effet de levier (QID, SDS, DXD). Chaque paire long-court suit un indice spécifique auquel ils offrent une exposition double: 
* S&P 500 avec SSO et SDS;
* NASDAQ-100 avec QLD et QID;
* Dow Jones Industrial Average avec DDM et DXD.

La formule pour le SSI est définie comme suit :
                
$$ {SSI}_t = \sum_{i \in J} \Delta_{i,t} - \sum_{i \in K} \Delta_{i,t} $$

où $$\Delta_{i,t} = SO_{i,t}/SO_{i,t-1} - 1$$ 
            
et $SO_{i,t}$, l'ETF outstanding shares à $T=t$.

                                
### Résultats Principaux
1. **Relation Négative avec les Retours Contemporains**: L'indice SSI est négativement corrélé avec les retours du marché contemporains, ce qui signifie qu'un sentiment spéculatif haussier apparaît généralement dans des marchés baissiers et inversement.
2. **Prédiction des Retours Futurs**: Le SSI prédit également négativement les retours futurs du marché. Une augmentation d'un écart-type du SSI est associée à une diminution de 1,14% à 1,67% des indices boursiers le mois suivant.
3. **Robustesse des Résultats**: Les résultats sont robustes même après avoir contrôlé pour d'autres proxies de sentiment et conditions de marché, et après des tests hors-échantillon.
4. **Analyse Empirique** : Les analyses régressives montrent que le SSI est un bon prédicteur des retours du marché. Le SSI capture des chocs de demande spéculative qui déplacent les prix du marché par rapport à leurs fondamentaux.

            
### Conclusion
L'indice ainsi crée permet :
1. **Mesure du Sentiment**: Le SSI fournit une mesure claire du sentiment spéculatif basé sur l'activité d'arbitrage.
2. **Prévisibilité des Retours**: Le SSI a une capacité prédictive significative pour les retours du marché, ce qui est économiquement pertinent pour les investisseurs.
3. **Distinction du Sentiment**: Le SSI est distinct des autres mesures de sentiment du marché et offre une nouvelle perspective sur la demande spéculative non fondamentale.
""")

elif onglet == "Construction":
    st.markdown(
        """
        <style>
        .subheader-text {
            color: orange;
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='subheader-text'>Construction de l'indice</h1>", unsafe_allow_html=True)

    ## Selection de la sources des données
    with st.form(key='form_1'):
        source = st.selectbox("Source des données",["Bloomberg","CSV File"])
        startDate = st.date_input("Choisissez une date", value=pd.to_datetime('2008-01-01'))
        endDate = st.date_input("Choisissez une date", value=pd.to_datetime('2023-01-01'))
        
        submit_button = st.form_submit_button(label='Run')

    if submit_button:
        if source == "CSV File" :
            SSI,data = building_SSI_csv()
        elif source =="Bloomberg" :
            SSI,data = building_SSI_bloom()
        data = filter_data(data,startDate,endDate)
        SSI = filter_data(SSI,startDate,endDate)
        
        ## Affichage
        st.markdown(" ")
        st.markdown('### Données des ETF')
        st.markdown("Voici le tableaux des données récupérée.")
        tickers = ['SDS',"QID","QLD","SSO","DDM","DXD"]
        st.dataframe(data)

        st.markdown(" ")
        st.markdown('### Graphes des ETF')
        st.markdown("Nous allons maintenant représenter l'allure des ETF à effet de levier utilisé pour la construction de l'indice sur des données allant de janvier 2008 jusqu'à avril 2023.\n")
        fig, axs = plt.subplots(3, 2, figsize=(15, 15)) 
        for i,ticker in enumerate(tickers):
            x = i%3
            y = i//3
            axs[x,y].plot(data.index,data[ticker])
            axs[x,y].set_title(f"Graphique de l'ETF {ticker}")
            axs[x,y].set_xlabel("Index")
            axs[x,y].set_ylabel("Price")
        st.pyplot(plt)

        st.markdown(" ")
        st.markdown("### Representation de l'indice SSI")
        fig, axs = plt.subplots(1, 1, figsize=(15, 15)) 
        axs.plot(SSI,marker='o')
        st.pyplot(plt)

elif onglet == "Analyses":
    st.markdown(
        """
        <style>
        .subheader-text {
            color: orange;
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='subheader-text'>Analyses</h1>", unsafe_allow_html=True)
    with st.form(key='form_1'):
        startDate = st.date_input("Choisissez une date", value=pd.to_datetime('2008-01-01'))
        endDate = st.date_input("Choisissez une date", value=pd.to_datetime('2023-01-01'))
        selection = st.selectbox("Variable de controle",["Pastor serie","Indice VIX","Investor Sentiment Index"])
        submit_button = st.form_submit_button(label='Run')
    if submit_button:
        SSI,data = building_SSI_csv()
        symbol = "^GSPC"
        SP500 = yf_Data.get_yf_datas(symbol,startDate,endDate)
        serie = SP500.reset_index(drop=False).resample('M', on='Date').mean().pct_change()*100
        serie.columns = ["SP500"]
        SSI_filter = filter_data(SSI,startDate,endDate)
        #Afficher SP500
        st.write(" ")
        st.write(" ")
        st.markdown("### Données SP500")
        st.markdown("Nous allons étudier le lien entre le SSI et le SP500. Commençons par étudier les données du SP500")
        fig, axs = plt.subplots(1, 1, figsize=(12,6)) 
        axs.plot(SP500)
        st.pyplot(plt)
        st.write("---")
        st.write(" ")

        st.markdown("### Analyse du SSI et Rendement globaux")
        st.markdown("#### 1 - Analyse comtemporaine")
        model,(x,y) = predictive_analysis_ssi(serie,SSI,0)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graphique de la Régression**")
            fig, ax = plt.subplots()
            ax.plot(y, label='Données initiales')
            ax.plot(model.predict(sm.add_constant(x)), color='red', label='Prédictions OLS')
            ax.legend()
            ax.set_xlabel('Dates')
            ax.set_ylabel('Rendements')
            ax.set_title('Rendement et prédictions des rendements en %')
            st.pyplot(plt)
        with col2:
            st.markdown("**Resultats Regression**")
            st.write(" ")
            st.write(model.summary2().tables[1])
            if model.pvalues["SSI"] < 0.05:
                st.markdown("La pvalue est inférieure à 0.05, c'est à dire le coefficient de SSI est signifiactif. Donc la régression comtemporaine montre que le SSI et l'indice ont une relation et que ainsi le SSI mesure les chocs de demande spéculative, qui déforment les prix des actions.")
            else :
                st.markdown("La pvalue est superieure à 0.05, c'est à dire le coefficient de SSI n'est pas signifiactif. Donc la régression comtemporaine montre que le SSI et l'indice ne sont pas forcément lié.")
  
        st.write(" ")
        st.markdown("#### 2 - Analyse prédictive")
        model,(x,y) = predictive_analysis_ssi(serie,SSI,-1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graphique de la Régression**")
            fig, ax = plt.subplots()
            ax.plot(y, label='Données initiales')
            ax.plot(model.predict(sm.add_constant(x)), color='red', label='Prédictions OLS')
            ax.legend()
            ax.set_xlabel('Dates')
            ax.set_ylabel('Rendements')
            ax.set_title('Rendement et prédictions des rendements en %')
            st.pyplot(plt)
        with col2:
            st.markdown("**Resultats Regression**")
            st.write(" ")
            st.write(model.summary2().tables[1])
            if model.pvalues["SSI"] < 0.05:
                st.markdown("La pvalue est inférieure à 0.05, c'est à dire le coefficient de SSI est signifiactif. Donc la régression prédictive entre le SSI en t et l'indice en t+1 montre que le SSI et l'indice sont lié. Ainsi le SSI qui mesure les chocs de demande spéculative déforment les prix des actions.")
            else :
                st.markdown("La pvalue est superieure à 0.05, c'est à dire le coefficient de SSI n'est pas signifiactif. Donc la régression prédictive entre le SSI en t et l'indice en t+1 montre que le SSI et l'indice ne sont pas forcément lié.")

        st.write(" ")
        st.markdown("#### 3 - Estimation du Coefficient par BootStrap")
        results = parametric_bootstrap(serie,SSI,100)
        st.write(results)
        st.markdown("Les biais sont proches de zéro et positifs, ce qui nous indique que les préoccupations d'estimation biaisée sont minimales.")
        if results["pvalue"].iloc[0] < 0.05:
                st.markdown("La pvalue est inférieure à 0.05, c'est à dire l'estimation n'est pas biaisée. La robustesse de notre conclusion est donc renforcée. Le SSI possède un pouvoir prédictif sur l'indice.")
        else :
                st.markdown("La pvalue est superieure à 0.05, c'est à dire l'estimation est peut-être biaisée.")

        
        ## Analyse Robustesse
        st.write("---")
        st.write(" ")
        st.markdown("### Analyse des perspectives économiques et robustesse")
        st.markdown("#### 1 - Sentiment de spéculation ou rééquilibrage rationnel")
        st.markdown("##### 1.1 - Regression prédictive sans SSI")
        model,(x,y) = predictive_analysis_serie(serie)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graphique de la Régression**")
            fig, ax = plt.subplots()
            ax.plot(y, label='Données initiales')
            ax.plot(model.predict(sm.add_constant(x)), color='red', label='Prédictions OLS')
            ax.legend()
            ax.set_xlabel('Dates')
            ax.set_ylabel('Rendements')
            ax.set_title('Rendement et prédictions des rendements en %')
            st.pyplot(plt)
        with col2:
            st.markdown("**Resultats Regression**")
            st.write(" ")
            st.write(model.summary2().tables[1])
            if model.pvalues[serie.columns[0]] < 0.1:
                st.markdown("La pvalue est inférieure à 0.1, c'est à dire le coefficient estimé est signifiactif. Donc les resultats montre que les rendements sont liés entre eux. Le SSI n'a donc pas forcement un pouvoir prédictif et peut seulement s'agir d'un rebalancement..")
            else :
                st.markdown("La pvalue est supérieure à 0.1, c'est à dire le coefficient estimé n'est pas signifiactif. Donc les resultats montre que les rendements ne sont pas liés entre eux. Le SSI est donc capable de prédir les rendements de notre indice et est okus qu'un simple rebalancement.")
        st.markdown("Nous allons donc voir comment change ces résultats quand on fait une regression multiple en ajoutant le SSI.")
        

        st.write("")
        st.write(" ")
        st.markdown("##### 1.2 - Regression prédictive avec SSI")
        model,(x,y) = predictive_analysis_serie_and_SSI(serie,SSI)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graphique de la Régression**")
            fig, ax = plt.subplots()
            ax.plot(y, label='Données initiales')
            ax.plot(model.predict(sm.add_constant(x)), color='red', label='Prédictions OLS')
            ax.legend()
            ax.set_xlabel('Dates')
            ax.set_ylabel('Rendements')
            ax.set_title('Rendement et prédictions des rendements en %')
            st.pyplot(plt)
        with col2:
            st.markdown("**Resultats Regression**")
            st.write(" ")
            st.write(model.summary2().tables[1])
            if model.pvalues["SSI"] < 0.1:
                st.markdown("La pvalue est inférieure à 0.1, c'est à dire le coefficient estimé est signifiactif. Donc les resultats montre que le SSI et l'indice en t ont une relation avec l'indice en t+1.")
            else :
                st.markdown("La pvalue est supérieure à 0.1, c'est à dire le coefficient estimé n'est pas signifiactif. Donc les resultats ne montrent pas une relation entre le SSI et les rendements anterieurs avec les rendements.")


        st.write(" ")
        st.markdown("#### 2 - Return Predictability Horizons")
        col1, col2 = st.columns(2)
        with col1:
            res = analyse_horizon(serie,SSI,7)
            st.write(res)
        with col2: 
            fig= plot_horizon(res,7)
            st.pyplot(plt)



        st.write(" ")
        st.markdown("#### 3 -  Return Predictability with Controls")
        if selection == "Pastor serie" :
            control = pd.read_csv("data/pastor_series.txt", delimiter='\t').reset_index(drop=False)
            control.columns = ['Date','Agg_Liq','Innov_Liq','Traded_Liq']
            control = control.set_index('Date')[['Agg_Liq']]
            control.index = pd.to_datetime(control.index,format='%Y%m')
            control=control.reset_index(drop=False).resample('M', on='Date').sum()
        elif selection == "Investor Sentiment Index" :
            control = pd.read_csv("data/investor_sentiment.csv", delimiter=';',index_col=0)[["SENT_ORTH"]]
            control.columns = ['investor_sentiment']
            control['investor_sentiment'] = control['investor_sentiment'].str.replace(",",'.').astype(float)
            control = control.rename_axis("Date", axis="index")
            control.index = pd.to_datetime(control.index,format='%Y%m')
            control=control.reset_index(drop=False).resample('ME', on='Date').sum()
        elif selection == "Indice VIX" :
            control = pd.read_excel("data/vix.xlsx",index_col=0)[['VIX Index']]
            control.index = pd.to_datetime(control.index)
            control=control.reset_index(drop=False).resample('M', on='Date').sum()
        control.columns = [selection]
        model,(x,y) = control_analysis(serie,SSI,control)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Graphique de la Régression**")
            fig, ax = plt.subplots()
            ax.plot(y, label='Données initiales')
            ax.plot(model.predict(sm.add_constant(x)), color='red', label='Prédictions OLS')
            ax.legend()
            ax.set_xlabel('Dates')
            ax.set_ylabel('Rendements')
            ax.set_title('Rendement et prédictions des rendements en %')
            st.pyplot(plt)
        with col2:
            st.markdown("**Resultats Regression**")
            st.write(" ")
            st.write(model.summary2().tables[1])
            st.markdown("**Corrélation entre le SSI et la variable de controle**")
            st.write(f"Corr(SSI,control) = ",x['SSI'].corr(x[selection]))
        if model.pvalues['SSI'] < 0.1:
            st.markdown("Le coefficient pour le SSI est significatif car p-value < 0.05, ce qui indique qu'il existe une relation négative entre le SSI et l'indice.")
        else :
            st.markdown("Le coefficient pour le SSI n'est pas significatif car p-value > 0.05, ce qui indique qu'il existe pas forcément une relation entre le SSI et l'indice.")
        if model.pvalues[selection] < 0.1:
            st.markdown("Le coefficient pour la variable de controle est significatif car p-value < 0.05, ce qui indique qu'il existe une relation entre la variable et l'indice.")
        else :
            st.markdown("Le coefficient pour la variable de controle n'est pas significatif car p-value > 0.05, ce qui indique qu'il existe pas forcément une relation entre la variable et l'indice.")
              
elif onglet == "Stratégie":
    st.markdown(
        """
        <style>
        .subheader-text {
            color: orange;
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='subheader-text'>Strategie</h1>", unsafe_allow_html=True)

    with st.form(key='form_1'):
        strategy_name = st.selectbox("Strategie",["Kalman Filter","Random Forest","Gated Recurrent Units"])
        startDate = st.date_input("Choisissez une date", value=pd.to_datetime('2006-01-01'))        
        endDate = st.date_input("Choisissez une date", value=pd.to_datetime('2023-01-01'))        
        frequence = st.selectbox("Frequence",["Mensuel","Hebdomadaire","Quotidien"])
        submit_button = st.form_submit_button(label='Run')

    if submit_button:
        map_freq={"Mensuel":"ME","Quotidien":"D","Hebdomadaire":"W"}
        frequence = map_freq[frequence]
        SSI,data = building_SSI_csv(frequence)
        SSI = filter_data(SSI,startDate,endDate)
        strategy = Trading_Strat(SSI)
        if strategy_name == "Kalman Filter":
            st.markdown(f"Nous allons appliquer une strategie avec un filtre de kalman permattant de debruiter la série et d'identifier nos signals d'achat ou de vente.")
            strategy.fn_Kalman_signal()
        elif strategy_name == "Random Forest":
            st.markdown(f"Nous allons appliquer une strategie avec un Random forest permettant de prédire nos signals d'achat ou de vente.")
            strategy.fn_RandomForest()
        elif strategy_name == "Gated Recurrent Units":
            st.markdown(f"Nous allons appliquer une strategie avec un Gated Recurrent Units permattant de prédire nos signals d'achat ou de vente.")
            strategy.fn_GRU()
        st.write("")
        st.markdown('### Données des signaux trouvés')
        st.write(strategy.df_Signal)
        st.write("")
        st.markdown("### Affichage des signaux")
        st.markdown("L'indice SSi va nous permettre à l'aide de notre filtre de kalman, d'identifer les moments d'achat ou de vente.")
        fig = strategy.plot_strat_signal()
        st.pyplot(plt)
        st.write("")
        st.markdown('### Données du Backtest')
        strategy.fn_backtest()
        st.write(strategy.df_Backtest)
        st.write("")
        st.markdown("### Affichage des rendements de la Stratégie")
        fig = strategy.plot_strat_return()
        st.pyplot(plt)

  