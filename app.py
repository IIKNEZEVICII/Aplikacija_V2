import streamlit as st
import tempfile
import os
import pandas as pd
import io
from logika_slike import poisci_in_izloci_listic, obarvaj_kapljice

st.set_page_config(layout="wide", page_title="Analiza WSP Lističev")
st.title("💧 Analiza vodno-občutljivih lističev")

# Inicializacija prostora za shranjevanje rezultatov in naključnega semena
if 'rezultati' not in st.session_state:
    st.session_state.rezultati = []
if 'roi_seed' not in st.session_state:
    st.session_state.roi_seed = 0

nalozena_datoteka = st.file_uploader("Naloži sliko lističa (JPG/PNG)", type=["jpg", "jpeg", "png"])

if nalozena_datoteka is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(nalozena_datoteka.read())
        tmp_pot = tmp.name

    original, izoliran = poisci_in_izloci_listic(tmp_pot)

    if izoliran is not None:
        st.markdown("### Analiza in obarvanje odtisov")
        
        col_nastavitve1, col_nastavitve2 = st.columns(2)
        with col_nastavitve1:
            obcutljivost = st.slider("Meja zaznavanja (Občutljivost)", 20, 230, 130, 5)
        with col_nastavitve2:
            prosojnost = st.slider("Prosojnost rdečih odtisov", 0.0, 1.0, 0.8, 0.05)
        
        # Obarvaj kapljice zdaj sprejme tudi roi_seed za naključno premikanje
        obarvan_listic, podatki_analize = obarvaj_kapljice(izoliran, prosojnost, obcutljivost, st.session_state.roi_seed)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(original, use_container_width=True)
        with col2:
            st.subheader("Analiziran listič (s 3x 1cm² ROI)")
            st.image(obarvan_listic, use_container_width=True)
            
        # Prikaz rezultatov in gumb za naključno premikanje območij
        st.markdown("---")
        col_naslov, col_premakni = st.columns([2, 1])
        with col_naslov:
            st.markdown("### Rezultati meritev (1 cm² območja)")
        with col_premakni:
            st.write("")
            if st.button("🔀 Naključno premakni območja"):
                st.session_state.roi_seed += 1
                st.rerun()

        c1, c2, c3 = st.columns(3)
        c1.metric("Območje 1 (Zgoraj)", f"{podatki_analize[0]['pokritost']}%", f"N = {podatki_analize[0]['stevilo']}")
        c2.metric("Območje 2 (Sredina)", f"{podatki_analize[1]['pokritost']}%", f"N = {podatki_analize[1]['stevilo']}")
        c3.metric("Območje 3 (Spodaj)", f"{podatki_analize[2]['pokritost']}%", f"N = {podatki_analize[2]['stevilo']}")

        # Orodje za dodajanje v tabelo
        st.markdown("---")
        col_input, col_btn = st.columns([3, 1])
        with col_input:
            oznaka = st.text_input("Oznaka vzorca za bazo (npr. 'zgoraj desno'):", value="")
        with col_btn:
            st.write("")
            st.write("")
            if st.button("➕ Shrani rezultat v bazo"):
                avg_p = round(sum([p['pokritost'] for p in podatki_analize]) / 3, 2)
                avg_n = round(sum([p['stevilo'] for p in podatki_analize]) / 3, 2)
                
                st.session_state.rezultati.append({
                    "Area 1 %": podatki_analize[0]['pokritost'],
                    "Area 1 N": podatki_analize[0]['stevilo'],
                    "Area 2 %": podatki_analize[1]['pokritost'],
                    "Area 2 N": podatki_analize[1]['stevilo'],
                    "Area 3 %": podatki_analize[2]['pokritost'],
                    "Area 3 N": podatki_analize[2]['stevilo'],
                    "Avg %": avg_p,
                    "Avg N": avg_n,
                    "Unnamed: 8": oznaka
                })
                # Pohranitvi ponastavimo seed nazaj na 0 za naslednji listič
                st.session_state.roi_seed = 0
                st.success(f"Dodan vzorec: {oznaka}")
    else:
        st.error("Algoritem ni zaznal lističa. Preverite osvetlitev.")

    os.remove(tmp_pot)

if len(st.session_state.rezultati) > 0:
    st.markdown("### 📊 Trenutna baza rezultatov")
    df = pd.DataFrame(st.session_state.rezultati)
    st.dataframe(df, use_container_width=True)
    
    col_ex1, col_ex2 = st.columns([2, 1])
    with col_ex1:
        ime_izvoza = st.text_input("Ime Excel datoteke:", value="3.blok-1D.xlsx")
    with col_ex2:
        st.write("")
        st.write("")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            ime_sheeta = ime_izvoza.replace(".xlsx", "")[:31]
            df.to_excel(writer, sheet_name=ime_sheeta, index=False)
            
        st.download_button(
            label="📥 Izvozi v Excel datoteko",
            data=buffer.getvalue(),
            file_name=ime_izvoza,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )