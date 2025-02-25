import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gaussian_kde

ExamplePats = pd.read_csv("med_events.csv", sep=',')

tidy = ExamplePats.copy()
tidy.columns = ["pnr", "eksd", "perday", "ATC", "dur_original"]
tidy['eksd'] = pd.to_datetime(tidy['eksd'], format='%m/%d/%Y')

def See(arg1,str1):
    C09CA01 = tidy[tidy['ATC'] == arg1].copy()

    Drug_see_p0 = C09CA01.copy()
    Drug_see_p1 = C09CA01.copy()

    Drug_see_p1 = Drug_see_p1.sort_values(by=['pnr', 'eksd']).groupby('pnr', group_keys=False)
    Drug_see_p1 = Drug_see_p1.apply(lambda x: x.assign(prev_eksd=x['eksd'].shift(1))).reset_index(drop=True)

    Drug_see_p1 = Drug_see_p1.dropna(subset=['prev_eksd'])

    Drug_see_p1 = Drug_see_p1.groupby('pnr', group_keys=False).apply(lambda x: x.sample(n=1)).reset_index(drop=True)
    Drug_see_p1 = Drug_see_p1[["pnr", "eksd", "prev_eksd"]].copy()
    Drug_see_p1['event.interval'] = (Drug_see_p1['eksd'] - Drug_see_p1['prev_eksd']).dt.days
    Drug_see_p1['event.interval'] = pd.to_numeric(Drug_see_p1['event.interval'])

    ecdf = ECDF(Drug_see_p1['event.interval'])
    x_ecdf = ecdf.x
    y_ecdf = ecdf.y

    if np.isneginf(x_ecdf[0]):
        x_ecdf = x_ecdf[1:]
        y_ecdf = y_ecdf[1:]

    ecdfs = [ECDF(group['event.interval'].astype(float).tolist()) for name, group in Drug_see_p1.groupby('pnr')]
    y_vals = [e(Drug_see_p1['event.interval']) for e in ecdfs]
    y = np.concatenate(y_vals)

    x = np.array(x_ecdf)
    x = pd.to_numeric(x)
    dfper = pd.DataFrame({'x': x, 'y': y_ecdf})


    dfper = dfper[dfper['y'] <= 0.8].copy() 

    max_x_dfper = dfper['x'].max()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dfper['x'], dfper['y'], label='80% ECDF')
    plt.title(f'80% ECDF ({str1})')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.plot(x_ecdf, y_ecdf, label='100% ECDF')
    plt.title(f'100% ECDF ({str1})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


    m1 = Drug_see_p1['pnr'].value_counts()
    plt.figure()
    plt.plot(m1.index, m1.values) 
    plt.title(f'Frequency of pnr ({str1})')
    plt.xlabel('pnr')
    plt.ylabel('Frequency')
    plt.show()


    ni = max_x_dfper
    Drug_see_p2 = Drug_see_p1[Drug_see_p1['event.interval'] <= ni].copy()

    if not Drug_see_p2['event.interval'].dropna().empty:
        vals = Drug_see_p2['event.interval'].astype(float).dropna()
        vals = vals[vals > 0]

        d = gaussian_kde(np.log(vals)) 
        
        log_vals = np.log(vals)
        x1 = np.linspace(log_vals.min(), log_vals.max(), 100) 
        y1 = d(x1)

        plt.figure()
        plt.plot(x1, y1)
        plt.title(f'Log(event interval) Density ({str1})')
        plt.xlabel('Log(event interval)')
        plt.ylabel('Density')
        plt.show()

        a = pd.DataFrame({'x': x1, 'y': y1})
        a_scaled = (a - a.mean()) / a.std() 

        if str1 == "Kmeans":
            silhouette_scores = []
            possible_clusters = range(2, 11) 

            for n_clusters in possible_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=1234, n_init=10) 
                cluster_labels = kmeans.fit_predict(a_scaled)
                silhouette_avg = silhouette_score(a_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)

            plt.figure()
            plt.plot(possible_clusters, silhouette_scores, marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette score")
            plt.title("Silhouette Analysis")
            plt.show()

            max_cluster_index = np.argmax(silhouette_scores)
            max_cluster = possible_clusters[max_cluster_index]


            kmeans_result = KMeans(n_clusters=max_cluster, random_state=1234, n_init=10) 
            cluster_labels_dfper = kmeans_result.fit_predict(dfper[['x']])
            dfper['cluster'] = cluster_labels_dfper + 1 
        elif str1 == "HDB":


            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1)
            cluster_labels_dfper = clusterer.fit_predict(dfper[['x']])

            dfper['cluster'] = np.where(cluster_labels_dfper == -1, 0, cluster_labels_dfper + 1)

        log_x_summary = dfper.groupby('cluster')['x'].apply(lambda x: pd.Series({
            'min': np.min(np.log(x)),
            'max': np.max(np.log(x)),
            'median': np.median(np.log(x))
        })).unstack()


        ni2 = pd.DataFrame({'Cluster': log_x_summary.index, 'Results': log_x_summary['min'].values})
        ni2['Results'] = ni2['Results'].replace([np.inf, -np.inf], 0) 

        ni3 = pd.DataFrame({'Cluster': log_x_summary.index, 'Results.1': log_x_summary['max'].values})
        ni3['Results.1'] = pd.to_numeric(ni3['Results.1'])

        nif = pd.concat([ni2.reset_index(drop=True), ni3['Results.1'].reset_index(drop=True)], axis=1)
        nif = nif.iloc[:, [0, 1, 2]] 
        nif.columns = ['Cluster', 'Results', 'Results.1'] 

        nif['Results'] = np.exp(nif['Results'])
        nif['Results.1'] = np.exp(nif['Results.1'])

        ni4 = pd.DataFrame({'Cluster': log_x_summary.index, 'Median': log_x_summary['median'].values})

        nif = pd.merge(nif, ni4, on='Cluster')
        nif.columns = ["Cluster", "Minimum", "Maximum", "Median"]
        nif['Median'] = nif['Median'].replace([np.inf, -np.inf], 0)
        nif = nif[nif['Median'] > 0].copy()


        results = pd.merge(Drug_see_p1.copy(), nif, how='cross') 
        results['Final_cluster'] = np.where((results['event.interval'] >= results['Minimum']) & (results['event.interval'] <= results['Maximum']), results['Cluster'], np.nan)
        results = results.dropna(subset=['Final_cluster']).copy() 
        results['Median'] = np.exp(results['Median'])
        results = results[["pnr", "Median", "Cluster"]].copy() 

        t1 = results['Cluster'].value_counts().sort_values(ascending=False).reset_index()
        t1.columns = ['Cluster', 'Freq']

        if t1.empty:
            t1_cluster_val = np.nan  
            t1 = pd.DataFrame({'Cluster': [np.nan]})
        else:
            t1_cluster_val = t1['Cluster'].iloc[0]
            t1 = pd.DataFrame({'Cluster': [t1_cluster_val]})
        

        t1['Cluster'] = pd.to_numeric(t1['Cluster'])
        results['Cluster'] = pd.to_numeric(results['Cluster'])

        t1_merged = pd.merge(t1, results, on="Cluster")
        if t1_merged.empty:

            t1 = pd.DataFrame({'Cluster': [np.nan], 'Median': [np.nan]})
        else:
            t1_merged = t1_merged.iloc[[0], :]
            t1_merged = t1_merged.drop(columns=['Freq'], errors='ignore')
            t1 = t1_merged.copy()


        Drug_see_p1_merged = pd.merge(Drug_see_p1.copy(), results, on="pnr", how='left', suffixes=('_drop', '')) 
        Drug_see_p1 = Drug_see_p1_merged.drop(columns=[col for col in Drug_see_p1_merged.columns if col.endswith('_drop')]) 

        t1_median_val = t1['Median'].iloc[0]
        Drug_see_p1['Median'] = Drug_see_p1['Median'].fillna(t1_median_val)
        Drug_see_p1['Cluster'] = Drug_see_p1['Cluster'].fillna(0).astype(str) 

        Drug_see_p1['event.interval'] = pd.to_numeric(Drug_see_p1['event.interval'])
        Drug_see_p1['test'] = round(Drug_see_p1['event.interval'] - Drug_see_p1['Median'], 1)

        for col in ['Median', 'Cluster']:
            if col not in Drug_see_p1.columns:
                Drug_see_p1[col] = np.nan

        Drug_see_p3 = Drug_see_p1[["pnr", "Median", "Cluster"]].copy()
    else: 
        for col in ['Median', 'Cluster']:
            if col not in Drug_see_p1.columns:
                Drug_see_p1[col] = np.nan

        Drug_see_p3 = Drug_see_p1[["pnr", "Median", "Cluster"]].copy()
        Drug_see_p3['Median'] = np.nan
        Drug_see_p3['Cluster'] = np.nan
        t1 = pd.DataFrame({'Cluster': [np.nan], 'Median': [np.nan]})


    Drug_see_p0_merged = pd.merge(Drug_see_p0.copy(), Drug_see_p3, on="pnr", how="left", suffixes=('_drop', '')) 
    Drug_see_p0 = Drug_see_p0_merged.drop(columns=[col for col in Drug_see_p0_merged.columns if col.endswith('_drop')]) 

    t1_median_val_final = t1['Median'].iloc[0] if not t1.empty and 'Median' in t1 else np.nan 
    Drug_see_p0['Median'] = pd.to_numeric(Drug_see_p0['Median']).fillna(t1_median_val_final) 
    Drug_see_p0['Cluster'] = pd.to_numeric(Drug_see_p0['Cluster'], errors='coerce').fillna(0).astype(int) 


    return Drug_see_p0


def see_assumption(arg1,str2):
    arg1 = arg1.sort_values(by=['pnr', 'eksd']).groupby('pnr')
    arg1 = arg1.apply(lambda x: x.assign(prev_eksd=x['eksd'].shift(1))).reset_index(drop=True)

    Drug_see2 = arg1.groupby('pnr').apply(lambda x: x.sort_values(by=['pnr', 'eksd']).assign(p_number=range(1, len(x) + 1))).reset_index(drop=True)
    Drug_see2 = Drug_see2[Drug_see2['p_number'] >= 2].copy()
    Drug_see2 = Drug_see2[["pnr", "eksd", "prev_eksd", "p_number"]].copy()
    Drug_see2['Duration'] = (Drug_see2['eksd'] - Drug_see2['prev_eksd']).dt.days
    Drug_see2['p_number'] = Drug_see2['p_number'].astype(str) 


    plt.figure()
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    plt.gca().set_title(f'Boxplot of Duration vs p_number ({str2})')
    plt.gca().set_xlabel('p_number')
    plt.gca().set_ylabel('Duration')
    plt.gca().set_facecolor('white') 
    plt.show()


    medians_of_medians = Drug_see2.groupby('pnr')['Duration'].median().median()


    plt.figure()
    sns.boxplot(x='p_number', y='Duration', data=Drug_see2)
    plt.axhline(y=medians_of_medians, color='red', linestyle='--')
    plt.gca().set_title(f'Boxplot of Duration vs p_number with Median of Medians ({str2})')
    plt.gca().set_xlabel('p_number')
    plt.gca().set_ylabel('Duration')
    plt.gca().set_facecolor('white') 
    plt.show()


Kmeans_medA = See("medA","Kmeans")
Kmeans_medB = See("medB","Kmeans")

HDB_medA = See("medA","HDB")
HDB_medB = See("medB","HDB")

see_assumption(Kmeans_medA,"Kmeans")
see_assumption(Kmeans_medB,"Kmeans")

see_assumption(HDB_medA,"HDB")
see_assumption(HDB_medB,"HDB")
