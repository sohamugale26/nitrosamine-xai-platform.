import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors
import py3Dmol
import streamlit.components.v1 as components

st.set_page_config(page_title="NitroPredict: Forced Degradation Engine", layout="wide")
st.title("NitroPredict: In-Silico Forced Degradation & Stress Testing")

if 'auto_ai_limit' not in st.session_state:
    st.session_state.auto_ai_limit = 1500.0  
if 'api_mw' not in st.session_state:
    st.session_state.api_mw = 300.0

tab1, tab2, tab3, tab4 = st.tabs(["🧬 API Structure", "💊 Formulation", "🔥 Stress Parameters", "📊 Degradation Kinetics"])

# --- TAB 1: 3D CHEMISTRY & STRUCTURE ---
with tab1:
    st.subheader("API Structural Vulnerability")
    
    portfolio = {
        "Custom Search (Live PubChem API)": "",
        "Paracetamol (Amide - Safe)": "CC(=O)NC1=CC=C(O)C=C1",
        "Metformin (Secondary/Guanidine)": "CN(C)C(=NC(=N)N)N",
        "Valsartan (Sartan Class)": "CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)C(C(C)C)C(=O)O"
    }
    selected_drug = st.selectbox("Select API:", list(portfolio.keys()), label_visibility="collapsed")
    
    final_smiles = ""
    if selected_drug == "Custom Search (Live PubChem API)":
        search_name = st.text_input("Enter API Name to fetch from NIH PubChem:")
        if search_name:
            import requests
            encoded_name = urllib.parse.quote(search_name.strip())
            with st.spinner(f"Querying NIH PubChem for '{search_name}'..."):
                try:
                    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        final_smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
                        st.success("✅ Live Connection Successful.")
                    else:
                        st.error("⚠️ Drug not found in PubChem.")
                except Exception:
                    st.error("Error connecting to external database.")
    else:
        final_smiles = portfolio[selected_drug]

    if final_smiles:
        molecule = Chem.MolFromSmiles(final_smiles)
        if molecule:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            mol_weight = Descriptors.MolWt(molecule)
            st.session_state.api_mw = mol_weight
            
            sec_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([C,c])([C,c])")
            tert_amine_pattern = Chem.MolFromSmarts("[NX3;H0;!$(NC=O)]([C,c])([C,c])([C,c])")
            
            has_sec = molecule.HasSubstructMatch(sec_amine_pattern)
            has_tert = molecule.HasSubstructMatch(tert_amine_pattern)
            
            st.info(f"**Formula:** {formula} | **Weight:** {mol_weight:.2f} g/mol")
            
            if has_sec:
                st.error("⚠️ **HIGH RISK:** Vulnerable Secondary (2°) Amine identified.")
                st.session_state.auto_ai_limit = 26.5
            elif has_tert:
                st.warning("⚠️ **MODERATE RISK:** Tertiary (3°) Amine identified.")
                st.session_state.auto_ai_limit = 400.0
            else:
                st.success("✅ **NO VULNERABLE AMINES DETECTED.**")
                st.session_state.auto_ai_limit = 1500.0
            
            molecule = Chem.AddHs(molecule)
            res = AllChem.EmbedMolecule(molecule, randomSeed=42)
            if res != 0: AllChem.Compute2DCoords(molecule)
            
            mblock = Chem.MolToMolBlock(molecule)
            view = py3Dmol.view(width=350, height=350)
            view.addModel(mblock, 'mol')
            view.setStyle({'stick': {}})
            view.zoomTo()
            components.html(view._make_html(), height=360, width=360)

# --- TAB 2: FORMULATION MASS ---
with tab2:
    st.subheader("API & Excipient Baseline Matrix")
    
    st.session_state.api_mass = st.number_input("API Mass per Unit (mg)", value=500.0, min_value=0.1)
    
    if 'formulation_df' not in st.session_state:
        st.session_state.formulation_df = pd.DataFrame({
            "Excipient": ["Microcrystalline Cellulose", "Magnesium Stearate", "Sodium Starch Glycolate"],
            "Amount (mg)": [100.0, 5.0, 20.0],
            "Nitrite Mean (ppm)": [2.5, 1.0, 0.8],
            "Nitrite StdDev (ppm)": [0.5, 0.2, 0.1]
        })
    st.session_state.edited_df = st.data_editor(st.session_state.formulation_df, num_rows="dynamic", use_container_width=True)

# --- TAB 3: STRESS PARAMETERS (FORCED DEGRADATION) ---
with tab3:
    st.subheader("Forced Degradation Testing Matrix")
    st.markdown("Configure extreme environmental parameters to force chemical breakdown (ICH Q1A).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 🧪 Stress Phase 1 (e.g., Acidic/Hydrolysis)")
        t1_c = st.slider("Stress Temperature (°C)", 40, 100, 80, key="t1_c")
        rh1 = st.slider("Chamber Humidity (%)", 0, 100, 75, key="rh1")
        time1 = st.number_input("Duration (Hours)", value=24.0, min_value=0.0, key="time1")
        ph_shift = st.slider("Acidic pH Shift (Intensity)", 0.0, 3.0, 1.5, help="Simulates 0.1N HCl exposure")
        
    with col2:
        st.markdown("##### 🔥 Stress Phase 2 (e.g., Thermal/Oxidative)")
        t2_c = st.slider("Stress Temperature (°C)", 40, 120, 100, key="t2_c")
        rh2 = st.slider("Chamber Humidity (%)", 0, 100, 20, key="rh2")
        time2 = st.number_input("Duration (Hours)", value=48.0, min_value=0.0, key="time2")

# --- TAB 4: DEGRADATION KINETICS ENGINE ---
with tab4:
    st.subheader("Stochastic ODE Degradation Solver")
    
    max_dose = st.number_input("Max Daily Dose of API (mg/day)", value=500.0, min_value=0.1)
    ppm_limit = st.session_state.auto_ai_limit / max_dose
    st.info(f"**Target Detection Limit for Chromatography:** {ppm_limit:.4f} ppm (ng/mg API)")
    
    if st.button("🚀 Run Forced Degradation Simulation", type="primary", use_container_width=True):
        
        df = st.session_state.edited_df
        if not final_smiles:
            st.error("🛑 Blocked: API Structure missing.")
            st.stop()
            
        with st.spinner("Integrating competing kinetic degradation pathways (5,000 Monte Carlo draws)..."):
            rng = np.random.default_rng(42)
            mc_draws = 5000
            
            excipient_mass_mg = df["Amount (mg)"].sum()
            total_mass_mg = excipient_mass_mg + st.session_state.api_mass
            
            base_nitrite_ppm = np.zeros(mc_draws)
            for idx, row in df.iterrows():
                if row["Amount (mg)"] > 0:
                    fraction = row["Amount (mg)"] / total_mass_mg
                    samples = rng.normal(row["Nitrite Mean (ppm)"], row["Nitrite StdDev (ppm)"], mc_draws)
                    base_nitrite_ppm += np.clip(samples, 0, None) * fraction
            
            # Synthetic parameter distributions for extreme stress
            Ea_loss = rng.normal(75.0, 3.0, mc_draws) * 1000
            Ea_form = rng.normal(85.0, 4.0, mc_draws) * 1000
            f_max = np.clip(rng.normal(0.15, 0.03, mc_draws), 0, 1) # Higher conversion cap under stress
            
            R = 8.314
            MW_nitrite = 46.0
            MW_NA = st.session_state.api_mw + 29.0 
            
            def calculate_rates(T_c, RH, Ea_l, Ea_f, ph_modifier=0):
                T_k = T_c + 273.15
                # Time basis shifted to HOURS for forced degradation
                # pH shift heavily amplifies the nitrosation formation constant
                k_loss = np.exp(21.0 - (Ea_l / (R * T_k)) + (0.02 * RH))
                k_form = np.exp(24.0 - (Ea_f / (R * T_k)) + (0.015 * RH) + (0.8 * ph_modifier))
                return k_loss, k_form
            
            final_NA_realistic = np.zeros(mc_draws)
            final_NA_conservative = np.zeros(mc_draws)
            
            for i in range(mc_draws):
                N0 = base_nitrite_ppm[i]
                cap = f_max[i] * N0 * (MW_NA / MW_nitrite) 
                
                # Phase 1 Kinetics (Includes Acidic Stress Modifier)
                k_l1, k_f1 = calculate_rates(t1_c, rh1, Ea_loss[i], Ea_form[i], ph_shift)
                # Phase 2 Kinetics (Thermal/Oxidative)
                k_l2, k_f2 = calculate_rates(t2_c, rh2, Ea_loss[i], Ea_form[i], 0)
                
                # REALISTIC ODE (Competing Degradation Pathways)
                k_tot1 = k_l1 + k_f1
                N_p1 = N0 * np.exp(-k_tot1 * time1)
                NA_p1 = (k_f1 * N0 / k_tot1) * (1 - np.exp(-k_tot1 * time1))
                
                k_tot2 = k_l2 + k_f2
                NA_final_real = NA_p1 + (k_f2 * N_p1 / k_tot2) * (1 - np.exp(-k_tot2 * time2))
                
                NA_final_real = NA_final_real * (MW_NA / MW_nitrite)
                final_NA_realistic[i] = min(NA_final_real, cap)
                
                # CONSERVATIVE ODE (Zero Benign Loss)
                NA_cons = N0 * (1 - np.exp(-k_f1 * time1)) 
                NA_cons += (N0 - NA_cons) * (1 - np.exp(-k_f2 * time2))
                final_NA_conservative[i] = NA_cons * (MW_NA / MW_nitrite)
                
                if final_NA_conservative[i] < final_NA_realistic[i]:
                    final_NA_conservative[i] = final_NA_realistic[i]
                    
            dilution_factor = total_mass_mg / st.session_state.api_mass
            final_NA_realistic *= dilution_factor
            final_NA_conservative *= dilution_factor
                
            p95_real = np.percentile(final_NA_realistic, 95)
            p95_cons = np.percentile(final_NA_conservative, 95)
            
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Predicted Yield (Realistic Competition)", f"{p95_real:.4f} ppm")
            col2.metric("Maximum Worst-Case Degradation", f"{p95_cons:.4f} ppm")
            
            st.markdown("### Method Development Conclusion")
            if p95_cons > ppm_limit:
                st.error(f"⚠️ **High Concentration Risk:** Under these forced stress conditions, predicted yields exceed the action limit of {ppm_limit:.4f} ppm. Your UPLC method must be strictly optimized to detect and quantify these elevated nitrosamine peaks without co-elution.")
            else:
                st.success(f"✅ **Low Susceptibility:** Even under extreme forced degradation, the theoretical yield ({p95_cons:.4f} ppm) fails to breach the regulatory action limit. The API exhibits high steric or chemical resistance to nitrosation.")
                
