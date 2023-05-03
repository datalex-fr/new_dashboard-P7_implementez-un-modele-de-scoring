import streamlit as st
import requests
import pandas as pd
import json
from pandas import json_normalize
import plotly.graph_objects as go
import seaborn as sns
from shap.plots import waterfall
import matplotlib.pyplot as plt
import shap

def main():

    API_URL = "https://oc-p7-api7.herokuapp.com/"

    st.set_page_config(page_title='Prêt à dépenser : P7',
                       layout='centered',
                       initial_sidebar_state='expanded')
    #titre dashboard
    st.title('Prêt à dépenser : P7')


    #fonctions
    def list_display_feature(f, def_n, key):
        all_feat = f
        n = st.slider("nombre de variable à afficher",
                      min_value=2, max_value=25,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'choisir les variables à afficher:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols

    #Endpoints
    #récuperer les ids client
    @st.cache(suppress_st_warning=True)
    def id_list():
        #creer l'url sk_id api
        id_api_url = API_URL + "id/"
        #requeter l'api et copier la reponse
        response = requests.get(id_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content)
        #recuperer les ids de "content"
        id_customers = pd.Series(content['data']).values
        return id_customers

    # Get selected customer's data ?????????????????????????????????????????
    data_type = []

    @st.cache
    def selected_client_data(selected_id):
        #creer l'url sk_id api
        data_api_url = API_URL + "data_client/?SK_ID_CURR=" + str(selected_id)
        #requeter l'api et copier la reponse
        response = requests.get(data_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content.decode('utf-8'))
        x_client = pd.DataFrame(content['data'])
        y_client = (pd.Series(content['y_client']).rename('TARGET'))
        return x_client, y_client

    @st.cache
    def get_all_cust_data():
        #creer l'url sk_id api
        data_api_url = API_URL + "all_proc_train_data/"
        #requeter l'api et copier la reponse
        response = requests.get(data_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content.decode('utf-8'))
        x_all_client = json_normalize(content['X_train']) #resultat des données
        y_all_client = json_normalize(content['y_train'].rename('TARGET')) #resultat des données
        return x_all_client, y_all_client

    #scoring
    @st.cache
    def fonction_score_model(selected_id):
        #creer l'url sk_id api
        score_api_url = API_URL + "score_du_client/?SK_ID_CURR=" + str(selected_id)
        #requeter l'api et copier la reponse
        response = requests.get(score_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content.decode('utf-8'))
        #recuperer les ids de "content"
        score_model = (content['score'])
        threshold = content['thresh']
        return score_model, threshold

    #liste des shap_values
    @st.cache
    def values_shap(selected_id):
        #creer l'url sk_id api
        shap_values_api_url = API_URL + "shap_values/?SK_ID_CURR=" + str(selected_id)
        #requeter l'api et copier la reponse
        response = requests.get(shap_values_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content)
        #recuperer les ids de "content"
        shapvals = pd.DataFrame(content['shap_val_client'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals


    #liste des expected values
    @st.cache
    def values_expect():
        #creer l'url sk_id api
        expected_values_api_url = API_URL + "exp_val/"
        #requeter l'api et copier la reponse
        response = requests.get(expected_values_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content)
        #recuperer les ids de "content"
        expect_vals = pd.Series(content['data']).values
        return expect_vals

    #list des noms des variables
    @st.cache
    def feature():
        #creer l'url sk_id api
        feat_api_url = API_URL + "feature/"
        #requeter l'api et copier la reponse
        response = requests.get(feat_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content)
        #recuperer les ids de "content"
        features_name = pd.Series(content['data']).values
        return features_name

    #liste des feature importances
    @st.cache
    def features_importances():
        #creer l'url
        feat_imp_api_url = API_URL + "feature_importance/"
        #requeter l'api et copier la reponse
        response = requests.get(feat_imp_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content.decode('utf-8'))
        #conversion en pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    #donnée des 10 nearest neighbors dans le train_set
    @st.cache
    def fonction_data_neigh_10(selected_id):
        #creer l'url scoring API
        neight_10_data_api_url = API_URL + "neigh_client_10/?SK_ID_CURR=" + str(selected_id)
        #requeter l'api et copier la reponse
        response = requests.get(neight_10_data_api_url)
        #conversion json en dictionnaire python
        content = json.loads(response.content.decode('utf-8'))
        #conversion en pd.DataFrame et pd.Series
        data_neig = pd.DataFrame(content['data_neigh'])
        target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        return data_neig, target_neig



    @st.cache
    def fonction_data_neigh_20(selected_id):
        neight_20_data_api_url = API_URL + "neigh_client_20/?SK_ID_CURR=" + str(selected_id)
        response = requests.get(neight_20_data_api_url)
        content = json.loads(response.content.decode('utf-8'))
        data_thousand_neig = pd.DataFrame(content['X_thousand_neigh'])
        x_custo = pd.DataFrame(content['x_custom'])
        target_thousand_neig = (pd.Series(content['y_thousand_neigh']).rename('TARGET'))
        return data_thousand_neig, target_thousand_neig, x_custo
    
    

    st.sidebar.title("menu")
    #liste des clients par ids
    cust_id = id_list()
    #selection de l'id
    selected_id = st.sidebar.selectbox('entrer un id:', cust_id, key=18)
    st.write('id selectionné = ', selected_id)

    #fonctions graphiques
    #shap global
    @st.cache
    def shap_summary():
        return shap.summary_plot(shap_vals, feature_names=features)

    #shap local
    @st.cache
    def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0, :],
                                                      max_display=nb, feature_names=ft)

    #shap local
    @st.cache(allow_output_mutation=True)
    def force_plot():
        shap.initjs()
        return shap.force_plot(expected_vals[0][0], shap_vals[0, :], matplotlib=True)

    #indicateur jauge
    @st.cache
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        if scor >= th:
            couleur_delta = 'red'
        elif scor < th:
            couleur_delta = 'Orange'

        if scor >= th:
            valeur_delta = "red"
        elif scor < th:
            valeur_delta = "green"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Selected Customer Score", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': valeur_delta}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 1.5, 'tickcolor': "black"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, int(th)], 'color': 'lightgreen'},
                    {'range': [int(th), int(scor)], 'color': couleur_delta}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 1,
                    'value': int(th)}}))

        fig.update_layout(paper_bgcolor="lavender", font={'color': "darkblue", 'family': "Arial"})
        return fig



    ### sidebar
    ### data
    if st.sidebar.checkbox("général"):
        st.markdown('données du client selectionné :')
        data_selected_cust, y_cust = selected_client_data(selected_id)
        st.write(data_selected_cust)
        
        
    ### ----------------------- prédiction d'un client ----------------
    
    if st.sidebar.checkbox("analyse & prédiction client", key=38):
        #récuperer score et le threshold
        score, threshold = fonction_score_model(selected_id)
        #afficher score et proba
        st.write('score probabilité : {:.0f}%'.format(score * 100))
        #afficher threshold
        st.write('threshold : {:.0f}%'.format(threshold * 100))
        #décision sur le pret bancaire
        if score >= threshold:
            decision = "crédit accordé"
        else:
            decision = "crédit rejeté"
        st.write("décision :", decision)
        
    ### ------------------- graph gauge ------------------------------
        figure = gauge_plot(score, threshold)
        st.write(figure)
        #coche
        st.markdown('jauge de score pour le client')
        expander = st.expander("explication classification lgbm =>")
        expander.write("la prédiction a été faite avec un model de classification binaire LGBM")
        expander.write("maximiser l'air sous la courbe ROC & FN et FP doivent être minimisés")

    ### ---------------- shap local -----------------------------------
        if st.checkbox('affichage diagramme shap', key=25):
            with st.spinner('graph SHAP en cours...............................................'):
                #récuprer valeurs shap pour le client & valeurs prédites
                shap_vals, expected_vals = values_shap(selected_id)
                #recuperer variable
                features = feature()
                nb_features = st.slider("nombre de variable à afficher",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                #graph
                waterfall_plot(nb_features, features, expected_vals[0][0], shap_vals.values)
                plt.gcf()
                st.pyplot(plt.gcf())
                #coche
                st.markdown('graph SHAP pour le client selectionné')
                #titre
                expander = st.expander("explication du SHAP =>")
                #explication
                expander.write("Le graphique en cascade ci-dessus affiche \
                explications pour la prédiction individuelle du client demandeur.\
                Le bas d'un tracé en cascade commence par la valeur attendue de la sortie du modèle \
                (c'est-à-dire la valeur obtenue si aucune information (caractéristiques) n'a été fournie), puis \
                chaque ligne montre comment la contribution positive (rouge) ou négative (bleue) de \
                chaque caractéristique déplace la valeur de la sortie de modèle attendue sur le \
                ensemble de données d'arrière-plan à la sortie du modèle pour cette prédiction.")

        ###------------------------ distribution ------------------------
        
        if st.checkbox('distribution', key=20):
            st.header('distribution variable principale')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('chargement distribution...'):
                #recupere les valeurs shap du client
                shap_vals, expected_vals = values_shap(selected_id)
                #recuperer les noms des variables
                features = feature()
                #recuperer les colonnes choisis
                disp_box_cols = list_display_feature(features, 2, key=45)
                # 10 neighbors du client :
                data_neigh, target_neigh = fonction_data_neigh_10(selected_id)
                data_thousand_neigh, target_thousand_neigh, x_customer = fonction_data_neigh_20(selected_id)
                x_cust, y_cust = selected_client_data(selected_id)
                x_customer.columns = x_customer.columns.str.split('.').str[0]
                target_neigh = target_neigh.replace({0: 'credit remboursé (10 voisins)',
                                                     1: 'default de paiement (10 voisins)'})
                
                target_thousand_neigh = target_thousand_neigh.replace({0: 'credit remboursé (20 voisins)',
                                                                       1: 'default de paiement (20 voisins)'})

                y_cust = y_cust.replace({0: 'credit remboursé',
                                         1: 'default de paiement'})



                # données client avec ses 10 neighbors
                df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",
                                                   value_name="values")

                sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                              palette=['darkgreen', 'darkred'], marker='o', size=15, edgecolor='k', ax=ax)

                
                
                # données client avec ses 20 neighbors
                df_thousand_neigh = pd.concat([data_thousand_neigh[disp_box_cols], target_thousand_neigh], axis=1)
                df_melt_thousand_neigh = df_thousand_neigh.reset_index()
                df_melt_thousand_neigh.columns = ['index'] + list(df_melt_thousand_neigh.columns)[1:]
                df_melt_thousand_neigh = df_melt_thousand_neigh.melt(id_vars=['index', 'TARGET'],
                                                                     value_vars=disp_box_cols,
                                                                     var_name="variables",  # "variables",
                                                                     value_name="values")

                sns.boxplot(data=df_melt_thousand_neigh, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)
                
                

                # client data
                df_selected_cust = pd.concat([x_customer[disp_box_cols], y_cust], axis=1)
                df_melt_sel_cust = df_selected_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")

                sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                              linewidth=1, color='y', marker='o', size=20,
                              edgecolor='k', label='applicant customer', ax=ax)

                # legende
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.write(fig)
                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.markdown('dispersion de la variable concerné avec\
                10 nearest neighbors')

                expander = st.expander("explication distribution =>")
                expander.write("les boîtes à moustaches montrent la dispersion des valeurs des caractéristiques prétraitées\
                utilisé par le modèle. La boîte à moustaches verte correspond aux clients qui ont remboursé \
                leur prêt, et les boîtes à moustaches rouges sont pour les clients qui ne l'ont pas remboursé.")


if __name__ == "__main__":
    main()
