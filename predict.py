import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix as cm
from preprocessing import pre_processing,  vektorisasi_teks, normalisasi_vektor, translate
import streamlit as st
import matplotlib.pyplot as plt
model = pickle.load(open('model_svc.pkl', 'rb'))
def predict(x_test):
    x_test = translate(x_test)
    x_test = pre_processing(x_test)
    x_test = vektorisasi_teks([x_test])
    x_test = normalisasi_vektor(x_test)
    predict = model.predict(x_test)
    predict_proba = model.predict_proba(x_test)
    return predict, predict_proba
def classification_report(x_test, y_test):
    predict = model.predict(x_test)
    akurasi = round(accuracy_score(y_test, predict),2)
    presisi = round(precision_score(y_test, predict),2)
    recall = round(recall_score(y_test, predict),2)
    f1 = round(f1_score(y_test, predict),2)
    return akurasi, presisi, recall, f1
def confusion_matrix(y_test, predict):
    confusion_matrix = cm(y_test, predict)
    fig = plt.figure(figsize = (10, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    st.pyplot(fig)
def hasil_predict(predict):
    if predict == 1:
        st.markdown('### Positif')
    else:
        st.markdown('### Negatif')