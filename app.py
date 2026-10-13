from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import streamlit as st
import pandas as pd
from predictor import Predictor

st.set_page_config(page_title="Cloudify | Monitoring", page_icon="⚡", layout="wide")
st.markdown('''<style>
.stApp {background:#101112;color:#f2f2e9}
.block-container {padding-top:2.4rem;max-width:1450px}
[data-testid="stSidebar"] {background:#17191b;border-right:1px solid #303236}
h1 {letter-spacing:-2px;font-size:3rem!important} h3 {letter-spacing:-.5px}
[data-testid="stMetric"] {background:#1b1d20;border:1px solid #303236;border-radius:14px;padding:20px}
[data-testid="stMetricLabel"] {color:#a6a9af}
[data-testid="stMetricValue"] {color:#f5d547;font-size:2rem}
.eyebrow {color:#f5d547;font-size:12px;letter-spacing:3px;font-weight:700}
.brand {font-size:30px;font-weight:800;letter-spacing:-1px;color:#f5d547}
.panel {border:1px solid #3b3d31;background:#24251b;padding:24px;border-radius:14px;margin-bottom:20px}
.muted {color:#a6a9af;font-size:14px;line-height:1.7}
.stButton>button[kind="primary"] {color:#101112;font-weight:700;border-radius:8px}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_predictor(dataset="google"):
    return Predictor(dataset)

with st.sidebar:
    st.markdown('<div class="brand">⚡ cloudify</div>', unsafe_allow_html=True)
    st.caption("CLOUD INTELLIGENCE")
    st.divider()
    page = st.radio("MONITORING", ["Scheduling", "Hardware"])
    st.divider()
    st.caption("PREDICTION ENGINE")
    model = st.selectbox("Primary model", ["Random Forest", "LSTM Autoencoder", "One-Class SVM"])
    st.markdown('<div class="muted">Random Forest + LSTM Autoencoder<br>One-Class SVM comparison baseline</div>', unsafe_allow_html=True)

try:
    engine = load_predictor()
except Exception as error:
    st.error("The prediction engine could not load. Install the pinned dependencies and confirm models/ contains the exported cloud artifacts.")
    st.exception(error)
    st.stop()

defaults = dict(scheduling_class=0, priority=100, collection_type=0, has_alloc=False,
                instance_index=100, cluster=3, vertical_scaling=1.0, scheduler=0.0,
                machine_bin=42, cycles_per_instruction=1.2, memory_accesses_per_instruction=0.008)
saved = st.session_state.setdefault("saved_inputs", defaults.copy())
def remember(key):
    saved[key] = st.session_state["_" + key]

for key, value in defaults.items():
    if (page == "Scheduling" and key not in ("cycles_per_instruction", "memory_accesses_per_instruction")) or (page == "Hardware" and key in ("cycles_per_instruction", "memory_accesses_per_instruction")):
        st.session_state["_" + key] = saved.get(key, value)
st.markdown('<div class="eyebrow">WORKSPACE / MONITORING</div>', unsafe_allow_html=True)
st.title(f"{page.lower()} intelligence.")
st.caption("Understand your workload. Evaluate task outcomes before you make your next move.")
st.write("")
a,b,c = st.columns(3)
a.metric("Engine", "Ready", "3 model algorithms", delta_color="off")
b.metric("Scope", "2 features", "Scheduling + Hardware", delta_color="off")
c.metric("Data mode", "Manual", "Scenario evaluation", delta_color="off")
st.write("")
left,right = st.columns([1.3,1], gap="large")
with left:
    st.subheader("Workload configuration" if page == "Scheduling" else "Hardware counters")
    st.caption("Changes are retained as you switch between Scheduling and Hardware.")
    if page == "Scheduling":
        st.info("Scheduling fields describe your workload. The parent models exclude these fields from inference; hardware counters drive the prediction.")
        x,y = st.columns(2)
        x.selectbox("Scheduling class", [0,1,2,3], format_func=lambda v:["Best effort", "Batch", "Standard", "Latency sensitive"][v], key="_scheduling_class",on_change=remember,args=("scheduling_class",))
        y.selectbox("Cluster", list(range(1,9)), key="_cluster",on_change=remember,args=("cluster",))
        st.slider("Priority", 0,450,key="_priority",on_change=remember,args=("priority",))
        x,y = st.columns(2)
        x.selectbox("Collection type", [0,1], format_func=lambda v:"Job" if v == 0 else "Allocation set", key="_collection_type",on_change=remember,args=("collection_type",))
        y.selectbox("Vertical scaling", [1.0,2.0,3.0], key="_vertical_scaling",on_change=remember,args=("vertical_scaling",))
        x.number_input("Instance index", min_value=0,max_value=177107,key="_instance_index",on_change=remember,args=("instance_index",))
        y.selectbox("Scheduler", [0.0,1.0], format_func=lambda v:"Default" if v == 0 else "Custom", key="_scheduler",on_change=remember,args=("scheduler",))
        st.checkbox("Reserved allocation assigned",key="_has_alloc",on_change=remember,args=("has_alloc",))
        st.slider("Machine ID bin",0,99,key="_machine_bin",on_change=remember,args=("machine_bin",),help="Machine identifier hashed into 100 bins; this is not a physical server ID.")
    else:
        st.markdown('<div class="panel"><b>Execution efficiency</b><div class="muted">Evaluate instruction stalls and memory intensity using the original model’s hardware counters.</div></div>',unsafe_allow_html=True)
        st.slider("Cycles per instruction (CPI)",0.1,10.0,step=0.01,key="_cycles_per_instruction",on_change=remember,args=("cycles_per_instruction",))
        st.caption("Higher values can indicate stalled or memory-bound execution.")
        st.slider("Memory accesses per instruction",0.0,0.1,step=0.0001,format="%.4f",key="_memory_accesses_per_instruction",on_change=remember,args=("memory_accesses_per_instruction",))
        st.divider()
        st.markdown("**Sequence evaluation · all three models**")
        st.caption("Upload 30–1000 consecutive SMD-format hardware observations. All three models evaluate the final observation, with LSTM using the last 30.")
        sample = Path(__file__).parent / "examples/hardware-sequence.csv"
        st.download_button("Download example CSV",sample.read_bytes(),"hardware-sequence.csv","text/csv")
        uploaded = st.file_uploader("Hardware sequence CSV",type=["csv"])
        sequence_run = st.button("Evaluate all three models",use_container_width=True)
        demo_run = st.button("Try bundled hardware example",use_container_width=True)
        if sequence_run or demo_run:
            try:
                if sequence_run and uploaded is None:
                    raise ValueError("Choose a CSV file first, or try the bundled example.")
                frame = pd.read_csv(sample if demo_run else uploaded)
                with st.spinner("Evaluating hardware sequence…"):
                    st.session_state["evaluation"] = load_predictor("smd").predict_sequence(frame)
                snapshot = {key: st.session_state.get("_" + key, saved[key]) for key in defaults}
                snapshot["priority_tier"] = next(i for i, bound in enumerate([89,115,199,359,450]) if snapshot["priority"] <= bound)
                st.session_state["evaluated_inputs"] = snapshot
                st.session_state["evaluation_mode"] = "SMD hardware sequence"
                st.session_state["evaluated_at"] = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%d %b %Y · %H:%M:%S WAT")
            except Exception as error:
                st.error(f"Sequence evaluation failed: {error}")
    st.write("")
    run = st.button("Evaluate manual hardware →",type="primary",use_container_width=True)

# Keep an independent copy: Streamlit removes widget keys when their page is hidden.
visible = list(defaults)[:9] if page == "Scheduling" else list(defaults)[9:]
for key in visible:
    saved[key] = st.session_state["_" + key]
values = saved.copy()
values["has_alloc"] = int(values["has_alloc"])
values["priority_tier"] = next(i for i, bound in enumerate([89,115,199,359,450]) if values["priority"] <= bound)
if run:
    try:
        st.session_state["evaluation"] = engine.predict(values)
        st.session_state["evaluated_inputs"] = values.copy()
        st.session_state["evaluation_mode"] = "Google manual hardware scenario"
        st.session_state["evaluated_at"] = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%d %b %Y · %H:%M:%S WAT")
    except Exception as error:
        st.error(f"Evaluation failed: {error}")

with right:
    st.subheader("Prediction overview")
    results = st.session_state.get("evaluation")
    if not results:
        st.markdown('<div class="panel"><div class="eyebrow">READY WHEN YOU ARE</div><h2>Your next decision,<br>in clearer focus.</h2><div class="muted">Evaluate manual hardware or open Hardware and try the bundled example to run all three models.</div></div>',unsafe_allow_html=True)
    else:
        if values != st.session_state["evaluated_inputs"]:
            st.warning("Inputs changed. Evaluate again to update these results.")
        selected = next((r for r in results if r["Model"] == model),results[0])
        st.caption(st.session_state["evaluation_mode"])
        if model == "LSTM Autoencoder" and len(results) == 2:
            st.info("LSTM requires a hardware sequence. Showing Random Forest for this manual scenario.")
        st.markdown(f'<div class="panel"><div class="eyebrow">{selected["Model"].upper()}</div><h2>{selected["Outcome"]}</h2><div class="muted">{st.session_state["evaluated_at"]}</div></div>',unsafe_allow_html=True)
        probability = selected["Failure probability"]
        if probability is not None:
            st.metric("Estimated failure probability", f"{probability:.1%}")
            st.progress(probability)
        else:
            st.metric("Anomaly score",f"{selected['Score']:.4f}")
            st.caption(f"Flagged above saved threshold {selected['Threshold']:.4f}. Anomaly scores are not failure probabilities.")
        st.caption(f"{sum(r['Outcome'] == 'FLAGGED' for r in results)} of {len(results)} models flag this observation.")
        frame = pd.DataFrame(results)
        frame["Failure probability"] = frame["Failure probability"].map(lambda p: f"{p:.1%}" if pd.notna(p) else "Not available")
        st.dataframe(frame,hide_index=True,use_container_width=True)
        st.download_button("Download evaluation",json.dumps({"evaluated_at":st.session_state["evaluated_at"],"inputs":st.session_state["evaluated_inputs"],"results":results},indent=2),"cloudify-evaluation.json","application/json")
    st.caption("Manual inputs use training medians for unexposed resource and usage fields. Sequence inputs use the parent SMD pipeline. No live cloud telemetry is connected.")
st.divider()
st.caption("CLOUDIFY  /  Scheduling & Hardware monitoring  ·  Powered by your trained cloud models")
