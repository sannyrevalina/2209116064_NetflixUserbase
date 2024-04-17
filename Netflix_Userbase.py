import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, chi2


sns.set()
sns.set_style("whitegrid")
sns.set_style("ticks",
    {"xtick.major.size":8,
     "ytick.major.size":8}
)


st.set_page_config(
    page_title="Netflix Userbase Analysis",
    page_icon="🎬",
)

st.title("Netflix Userbase Analysis")

page = st.sidebar.selectbox("Select Page", ["Bussines scenario", "Visualisasi Data", "Evaluasi Data"])

if page == "Bussines scenario":
    st.image("dataset-cover.png")
    st.write("tujuan dari dataset ini adalah untuk mengetahui tingkat penggunaan langganan pada aplikasi Netflix.pada dataset ini terdapat informasi seperti jenis langganan pengguna (Basic, Standard, atau Premium), pendapatan bulanan yang dihasilkan dari langganan mereka, tanggal mereka bergabung dengan Netflix (Tanggal Bergabung), tanggal pembayaran terakhir mereka (Tanggal Pembayaran Terakhir), dan negara tempat mereka berada.")
    st.write("tujuan dari data mining ini adalah untuk memprediksi pengguna Netflix kedepannya berdarsakan faktor-faktor yang mempengaruhi data ini akan dianalisis menggunakan berbagai metode analisis data untuk mengungkap pola dan hubungan yang dapat memberikan wawasan mendalam tentang perilaku pengguna Netflix.")


elif page == "Visualisasi Data":
    
    df = pd.read_csv("Netflix Userbase.csv")

    st.header("Visualisasi Data")

    view_option = st.selectbox("View Data:", ["Tampilan berdasarkan usia", "Tipe langganan", "Tipe langganan berdasarkan jenis kelamin"])

    if view_option == "Tampilan berdasarkan usia":
        plt.figure(figsize=(8, 6))
        sns.histplot(df['Age'].dropna(), bins=10, kde=True)
        plt.ylabel('jumlah pengguna')
        plt.xlabel('Age')
        st.pyplot(plt.gcf())  
        st.write('Dari histogram di atas, terlihat bahwa sebagian besar pengguna Netflix berada dalam rentang umur tertentu. Distribusi umur tidak berbentuk normal Berdasarkan pola distribusi umur yang terlihat, mungkin ada peluang untuk meningkatkan penawaran konten atau fitur yang lebih menarik bagi kelompok usia tertentu, dengan ada nya fitur-fitur baru diharapkan ada nya peningkatan dalam penggunaan netflix.')

    elif view_option == "Tipe langganan":
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Subscription Type', data=df, palette='Set2')
        plt.title('Tipe layanan')
        plt.xlabel('Tipe layanan')
        plt.ylabel('jumlah pengguna')
        st.pyplot(plt.gcf()) 
        st.write('dari visualisasi diatas ini dapat diketahui bahwa tipe yang paling banyak digunakan oleh user adalah basic')

    elif view_option == "Tipe langganan berdasarkan jenis kelamin":
        g = sns.FacetGrid(df, col="Gender", row="Subscription Type", margin_titles=True)
        g.map(plt.hist, "Age", color="steelblue")
        plt.xlabel('usia')
        plt.ylabel('jumlah pengguna')
        st.pyplot(g.fig)  
        st.write('pada visualisasi diatas terbagi menjadi 3 tipe langganan, yaitu basic, premium dan standar. pada gambar diatas merupakan perbandingan antara pengguna netflix perempuan dan laki-laki berdasarkan umur dan tipe langganan yang digunakan.')


    pie_chart_option = st.selectbox("tampilan diagram lingkaran:", ["Distribusi Gender", "Distribusi Tipe Langganan", "Distribusi Perangkat"])


    if pie_chart_option == "Distribusi Gender":
        gender_counts = df['Gender'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribusi Gender')
        plt.axis('equal') 
        st.pyplot(plt.gcf())  
        st.write('dari visualisasi pie chart diatas dapat diketahui bahwa penguna netflix yang paling banyak adalah perempuan yaitu sebesar 50.3% sedangkan laki-laki sebesar 49.7%.')

    elif pie_chart_option == "Distribusi Tipe Langganan":
        SubscriptionType_counts = df['Subscription Type'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(SubscriptionType_counts, labels=SubscriptionType_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribusi Tipe Langganan')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('dari visualisasi diatas terdapat 3 jenis tipe pengguna yaitu basic,standar dan premium. pada gambar diatas dapat diketahui bahwa tipe langganan yang palling banyak digunakan adalah basic yaitu sebesar 40.0%, selanjutnya adalah standard yaitu sebesar 30.7%, lalu premium sebesar 29.3%.')


    elif pie_chart_option == "Distribusi Perangkat":
        Device_counts = df['Device'].value_counts()
        plt.figure(figsize=(6, 6))
        plt.pie(Device_counts, labels=Device_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Distribusi Perangkat')
        plt.axis('equal')  
        st.pyplot(plt.gcf())  
        st.write('pada visualisasi diatas dapat diketahui banyak device yang digunakan user untuk mengakses netflix ada 4 perangkat yang dapat digunakan, gambar diatas juga memberikan informasi mengenai presentase perangkat yang digunakan. dari gambar tersebut dapat diketahui bahwa perangkat yang paling banyak digunakan adalah laptop yaitu sebesar 25.4%.')

    st.subheader('Number of Missing Values for Each Column:')
    missing_values = df.isnull().sum()
    st.write(missing_values)
    st.write('dari hasil diatas dapat disimpulkan bahwa setiap kolom pada dataset memiliki nilai/lengkap(not null), dikarenakan jumlah nilai kosong untuk setiap kolom adalah 0.')

    st.subheader('Outliers Values:')
    sns.boxplot(x='Subscription Type', y='Monthly Revenue', data=df, palette='pastel')
    plt.show()
    st.pyplot(plt.gcf()) 
    st.write('dari visualisasi diatas tidak terdapat outlier pada setiap jenis tipe langganan terhadap pendapatan bulanan, dari hasil ini dapat disimpulkan bahwa mayoritas pendapatan bulanan berada dalam kisaran yang serupa di setiap kategori langganan (basic, premium, dan standard).')


    df = pd.read_csv("Netflix Userbase.csv")

   
    def kategori(age):
        if 26 <= age < 49:
            return 'Adult'
        else:
            return 'Elderly'

    df['Kategori'] = df['Age'].apply(kategori)

    st.write(df.head())

    
    df['Join Date'] = pd.to_datetime(df['Join Date'])
    df['Last Payment Date'] = pd.to_datetime(df['Last Payment Date'])
    df['Durasi Langganan (hari)'] = (df['Last Payment Date'] - df['Join Date']).dt.days

    st.write(df.head())

    st.subheader('Data Reduction')
    numeric_columns_before = df.select_dtypes(include=['number']).columns
    df_numeric_before = df[numeric_columns_before]
    df_corr_before = df_numeric_before.corr()

      
    fig_before = px.imshow(df_corr_before)
    st.plotly_chart(fig_before)
    st.write('tampilan diatas merupakan matriks kolerasi dalam bentuk heatmap. tampilan diatas merupakan tampilan pada heatmap sebelum user id dihapus')

     
    df = df.drop(['User ID'], axis=1)

    numeric_columns_after = df.select_dtypes(include=['number']).columns
    df_numeric_after = df[numeric_columns_after]
    df_corr_after = df_numeric_after.corr()

    fig_after = px.imshow(df_corr_after)
    st.plotly_chart(fig_after)
    st.write('gambar diatas merupakan tampilan setelah user id dihapus')
        
    st.subheader('Data Transformation')
  
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Kategori'] = df['Kategori'].map({'Adult': 1, 'Elderly': 2}).astype(int)
    device_mapping = {'Laptop': 1, 'Tablet': 2, 'Smartphone': 3, 'Smart TV': 4, 'Unknown': 0}
    df['Device'] = df['Device'].map(device_mapping).fillna(0).astype(int)
    subscription_mapping = {'Basic': 0, 'Premium': 1, 'Standard': 2, 'Unknown': -1}
    df['Subscription Type'] = df['Subscription Type'].map(subscription_mapping).fillna(-1).astype(int)

    st.write("Tampilan tabel (Mapping):")
    st.write(df.head())
    st.write('data di atas merupakan tampilan dari data yang sudah diubah.pada kolom ini terdapat kolom gender yang berisikan int 0 dan 1 dimana 0 merupakan female dan 1 merupakan male. selain itu terdapat juga kolom device yang berisikan int 1,2,3,4. dimana 1 merupakan kode dari Device Laptop, 2 adalah tablet, 3 adalah Smartphone dana 4 adalah smart tv. ') 

    st.write("Encoding:")
    df = pd.get_dummies(df)
    st.write(df.head())

    df_cleaned = pd.read_csv("Data Cleaned.csv")

    st.write("Encoding cleaned data:")
    df_cleaned_encoded = pd.get_dummies(df_cleaned)
    st.write(df_cleaned_encoded.head())
    df.to_csv("Data Cleaned.csv", index=False)
    df_data= pd.read_csv("Data Cleaned.csv")
        
elif page == "Evaluasi Data":
    st.header("Evaluation")
    
    df_cleaned_encoded = pd.read_csv("Data Cleaned.csv")

    
    x = df_cleaned_encoded.drop('Device', axis=1)
    y = df_cleaned_encoded['Device']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import MinMaxScaler

    non_numeric_columns = x_train.select_dtypes(exclude=['float64', 'int64']).columns

    x_train_numeric = x_train.drop(columns=non_numeric_columns)
    x_test_numeric = x_test.drop(columns=non_numeric_columns)

    scaler = MinMaxScaler()

    x_train_norm = scaler.fit_transform(x_train_numeric)

    x_test_norm = scaler.transform(x_test_numeric)

    #Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(x_train_norm, y_train)

    #K-Nearest Neighbor
    knn = KNeighborsClassifier()
    knn.fit(x_train_norm, y_train)
    
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train_norm, y_train)

    gnb_pred = gnb.predict(x_test_norm)
    knn_pred = knn.predict(x_test_norm)
    dtc_pred = dtc.predict(x_test_norm)

    x_test = pd.DataFrame(x_test).reset_index(drop=True)

    y_test = pd.DataFrame(y_test).reset_index(drop=True)

    gnb_col = pd.DataFrame(gnb_pred.astype(int), columns=["gnb_prediction"])
    knn_col = pd.DataFrame(knn_pred.astype(int), columns=["knn_prediction"])
    dtc_col = pd.DataFrame(dtc_pred.astype(int), columns=["dtc_prediction"])

    combined_data = pd.concat([x_test, y_test, gnb_col, knn_col, dtc_col], axis=1)

    st.write('tampilan tabel combined data')
    st.write(combined_data.head(5))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    st.subheader(' Klasifikasi')

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,6))

    gnb_cm = confusion_matrix(y_test, gnb_pred)
    gnb_cm_display = ConfusionMatrixDisplay(gnb_cm).plot(ax=axes[0], cmap='inferno')
    gnb_cm_display.ax_.set_title("Gaussian Naive Bayes")

    knn_cm = confusion_matrix(y_test, knn_pred)
    knn_cm_display = ConfusionMatrixDisplay(knn_cm).plot(ax=axes[1], cmap='inferno')
    knn_cm_display.ax_.set_title("K-Nearest Neighbor")

    dtc_cm = confusion_matrix(y_test, dtc_pred)
    dtc_cm_display = ConfusionMatrixDisplay(dtc_cm).plot(ax=axes[2], cmap='inferno')
    dtc_cm_display.ax_.set_title("Decision Tree Classifier")

    st.pyplot(fig)  
    st.write('pada gambar diatas terdapat tampilan dari 3 confusion matrix yaitu Gaussian Naive Bayes, K-Nearest Neighbor, dan Decision Tree Class. kolom tersbut memiliki ukuran 4x4 karena jumlah data pada kolom yang saya gunakan berjumlah 4 yang menunjukkan ada empat kelas yang diamati.')
    import numpy as np
    from sklearn.metrics import roc_auc_score

    gnb_cm = np.array(gnb_cm)

    
    tn = gnb_cm[0, 0]
    fp = np.sum(gnb_cm[0, 1:])  # Jumlah semua nilai pada baris pertama kecuali elemen di diagonal utama
    fn = np.sum(gnb_cm[1:, 0])  # Jumlah semua nilai pada kolom pertama kecuali elemen di diagonal utama
    tp = np.sum(gnb_cm[1:, 1:])  # Jumlah semua nilai pada matriks konfusi kecuali baris dan kolom pertama

    # Perhitungan metrik evaluasi
    accuracy = (tp + tn) / np.sum(gnb_cm)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)

    # Menghitung ROC AUC Score
    y_true = np.array([0] * tn + [1] * fp + [1] * fn + [1] * tp) 
    y_score = np.array([0] * (tn + fp) + [1] * (fn + tp))  
    roc_score = roc_auc_score(y_true, y_score)

    st.write("Nilai Akurasi: ", accuracy)
    st.write("Nilai Presisi: ", precision)
    st.write("Nilai Recall: ", recall)
    st.write("Nilai F1-Score: ", f1_score)
    st.write("Nilai ROC AUC Score: ", roc_score)
    st.write('pada output diatas terdapat nilai akurasi sebesar 0.576 atau sebesar 57,6%, nilai presisi yang bergunana untuk mengukur seberapa sering prediksi positif model, pada output diatas nilai presisi adalah sebesar 0.74% yang memiliki arti semua prediksi positif yang dibuat oleh model bernilai benar sebanyak 0.74%. nilai recall pada output digunakan untuk mendeteksi kelas yang sebenarnya bernilai positif. f1 merupakan nilai rata-rata dari presisi dan recall, dan nilai roc berguna untuk mengukur kinerja model')

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

   
    X_train, X_test, y_train_r, y_test_r = train_test_split(data, target, test_size=0.2, random_state=42)

   
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train_r)
    y_pred_lr = model_lr.predict(X_test)

    
    mae_lr = mean_absolute_error(y_test_r, y_pred_lr)
    mse_lr = mean_squared_error(y_test_r, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mape_lr = mean_absolute_percentage_error(y_test_r, y_pred_lr)

 
    model_dt = DecisionTreeRegressor()
    model_dt.fit(X_train, y_train_r)
    y_pred_dt = model_dt.predict(X_test)

   
    mae_dt = mean_absolute_error(y_test_r, y_pred_dt)
    mse_dt = mean_squared_error(y_test_r, y_pred_dt)
    rmse_dt = np.sqrt(mse_dt)
    mape_dt = mean_absolute_percentage_error(y_test_r, y_pred_dt)


    df_eval = pd.DataFrame({'Model': ['Linear Regression', 'Decision Tree'],
                            'MAE': [mae_lr, mae_dt],
                            'MSE': [mse_lr, mse_dt],
                            'RMSE': [rmse_lr, rmse_dt],
                            'MAPE': [mape_lr, mape_dt]})

    st.title('evaluasi untuk model regresi')
    st.write(df_eval)
    st.write('pada output diatas terdapat MAE, MSE,RMSE,DAN MAPE. MAE digunakan untuk mengukur rata-rata dari selisih absolut antara prediksi dan nilai sebenarnya. Semakin rendah nilai MAE, semakin baik model dalam membuat prediksi yang akurat.')
    from sklearn.metrics import roc_auc_score, roc_curve
    models = [knn, gnb, dtc]
    model_names = ['K-Nearest Neighbor', 'Gaussian Naive Bayes', 'Decision Tree Classifier']

    st.title('ROC Curve untuk klasifikasi model')
    st.subheader('Receiver Operating Characteristic (ROC) Curve')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for model, name, ax in zip(models, model_names, axes):
        y_pred = model.predict(x_test_norm)

        y_test_binary = (y_test == 1).astype(int)

        fpr, tpr, _ = roc_curve(y_test_binary, y_pred)

       
        roc_auc = roc_auc_score(y_test_binary, y_pred)

   
        ax.plot(fpr, tpr, label=f'{name} (ROC-AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {name}')
        ax.legend(loc='lower right')

    
    st.pyplot(fig)
    st.write('pada output cross validation diatas dapat disimpulkan bahwa ROC Curve GNB merupakan model terbaik dalam membedakan data positif dan negatif, diikuti oleh KNN dan DTC.')
    from sklearn.model_selection import cross_val_score

  
    models = [gnb, knn, dtc]
    model_names = ['Gaussian Naive Bayes', 'K-Nearest Neighbors', 'Decision Tree Classifier']

   
    cv_scores = []
    for model in models:
        scores = cross_val_score(model, x_train_norm, y_train, cv=5)
        cv_scores.append(scores)

  
    df_cv_scores = pd.DataFrame(cv_scores, index=model_names).T

    
    st.title('Cross-Validation Scores for Different Models')
    st.subheader('Accuracy Across Folds')

   
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_cv_scores, markers=True, ax=ax)
    ax.set_title('Cross-Validation Scores for Different Models')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(title='Model', loc='lower right')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])

    st.pyplot(fig)
    
    from sklearn.feature_selection import SelectKBest, chi2
    selector = SelectKBest(score_func=chi2, k=6)

    selector.fit(x_train_norm, y_train)

    
    selected_indices = selector.get_support(indices=True)

    
    selected_features = x_train.columns[selected_indices]

    selected_ranks = selector.scores_[selected_indices]

    feature_ranks_df = pd.DataFrame({'Feature': selected_features, 'Rank': selected_ranks})

    
    feature_ranks_df = feature_ranks_df.sort_values(by='Rank', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_ranks_df['Feature'], feature_ranks_df['Rank'], color='skyblue')
    plt.xlabel('Rank')
    plt.ylabel('Feature')
    plt.title('Feature Ranking')
    plt.gca().invert_yaxis()

   
    st.pyplot(plt)

