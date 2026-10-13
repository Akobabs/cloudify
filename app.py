from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import time
import streamlit as st
import pandas as pd
from predictor import Predictor

ROOT = Path(__file__).parent
st.set_page_config(page_title='Cloudify | Monitoring', page_icon='⚡', layout='wide')
st.markdown('''<style>
.stApp{background:#101112;color:#f2f2e9}.block-container{padding-top:2.4rem;max-width:1450px}
[data-testid="stSidebar"]{background:#17191b;border-right:1px solid #303236}
h1{letter-spacing:-2px;font-size:3rem!important}[data-testid="stMetric"]{background:#1b1d20;border:1px solid #303236;border-radius:14px;padding:18px}
[data-testid="stMetricValue"]{color:#f5d547;font-size:2rem}.eyebrow{color:#f5d547;font-size:12px;letter-spacing:3px;font-weight:700}
.brand{font-size:30px;font-weight:800;color:#f5d547}.panel{border:1px solid #3b3d31;background:#24251b;padding:22px;border-radius:14px;margin-bottom:16px}
.muted{color:#a6a9af;font-size:14px;line-height:1.7}.stButton>button[kind="primary"]{color:#101112;font-weight:700}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def load_predictor(dataset='google'):
    return Predictor(dataset)

try:
    engine = load_predictor()
except Exception as error:
    st.error('Could not load the bundled models. Deploy using Python 3.12 and the pinned dependencies.')
    st.exception(error)
    st.stop()

FIELDS = {
    'cycles_per_instruction': ('Cycles per instruction (CPI)', 10.0, .01),
    'memory_accesses_per_instruction': ('Memory accesses per instruction', .1, .0001),
    'average_cpu': ('Average CPU', 1.0, .001),
    'maximum_cpu': ('Peak CPU', 1.0, .001),
    'sample_cpu': ('Sample CPU', 1.0, .001),
    'requested_cpu': ('Requested CPU', 1.0, .001),
    'average_memory': ('Average memory', 1.0, .001),
    'maximum_memory': ('Peak memory', 1.0, .001),
    'requested_memory': ('Requested memory', 1.0, .001),
    'assigned_memory': ('Assigned memory', 1.0, .001),
    'page_cache_memory': ('Page cache memory', 1.0, .0001),
}
medians = dict(zip(engine.preprocessor['raw_features'], engine.preprocessor['imputer'].statistics_))
defaults = {key: float(medians[key]) for key in FIELDS}
saved = st.session_state.setdefault('hardware_inputs', defaults.copy())
st.session_state.setdefault('history', [])
st.session_state.setdefault('manual_revision', 0)

def remember(key):
    st.session_state['hardware_inputs'][key] = st.session_state['input_' + key]

def reset_readings():
    st.session_state['hardware_inputs'] = defaults.copy()
    st.session_state['manual_revision'] += 1

with st.sidebar:
    st.markdown('<div class="brand">⚡ cloudify</div>', unsafe_allow_html=True)
    st.caption('CLOUD INTELLIGENCE')
    st.divider()
    page = st.radio('MONITORING', ['Scheduling', 'Hardware'], index=1, key='navigation')
    model = st.selectbox('Primary model', ['Random Forest', 'LSTM Autoencoder', 'One-Class SVM'])
    st.divider()
    st.caption('Random Forest + LSTM Autoencoder\n\nOne-Class SVM comparison baseline')
    st.caption('Automatic inference · user-entered telemetry')

st.markdown('<div class="eyebrow">WORKSPACE / MONITORING</div>', unsafe_allow_html=True)
st.title(f'{page.lower()} intelligence.')
st.caption('Updated readings. Immediate inference. A clear record of each evaluation.')

if page == 'Scheduling':
    st.subheader('Workload configuration')
    st.info('Scheduling settings describe your workload. These fields are excluded from the parent models; change Hardware readings to update predictions.')
    scheduling = st.session_state.setdefault('scheduling', {'scheduling_class':0,'priority':100,'cluster':3,'instance_index':100,'collection_type':0,'vertical_scaling':1.0,'scheduler':0.0,'has_alloc':False,'machine_bin':42})
    def save_schedule(key):
        scheduling[key] = st.session_state['schedule_' + key]
    for key,value in scheduling.items():
        st.session_state['schedule_' + key] = value
    a,b = st.columns(2)
    a.selectbox('Scheduling class',[0,1,2,3],format_func=lambda v:['Best effort','Batch','Standard','Latency sensitive'][v],key='schedule_scheduling_class',on_change=save_schedule,args=('scheduling_class',))
    b.selectbox('Cluster',list(range(1,9)),key='schedule_cluster',on_change=save_schedule,args=('cluster',))
    st.slider('Priority',0,450,key='schedule_priority',on_change=save_schedule,args=('priority',))
    a.number_input('Instance index',min_value=0,max_value=177107,key='schedule_instance_index',on_change=save_schedule,args=('instance_index',))
    b.selectbox('Collection type',[0,1],format_func=lambda v:'Job' if v==0 else 'Allocation set',key='schedule_collection_type',on_change=save_schedule,args=('collection_type',))
    a.selectbox('Vertical scaling',[1.0,2.0,3.0],key='schedule_vertical_scaling',on_change=save_schedule,args=('vertical_scaling',))
    b.selectbox('Scheduler',[0.0,1.0],format_func=lambda v:'Default' if v==0 else 'Custom',key='schedule_scheduler',on_change=save_schedule,args=('scheduler',))
    st.checkbox('Reserved allocation assigned',key='schedule_has_alloc',on_change=save_schedule,args=('has_alloc',))
    st.slider('Machine ID bin',0,99,key='schedule_machine_bin',on_change=save_schedule,args=('machine_bin',))
    st.download_button('Download workload configuration',json.dumps(scheduling,indent=2),'cloudify-workload.json','application/json')
    st.caption('Configuration is retained within this browser session. It does not dispatch jobs.')
else:
    mode = st.radio('Reading source',['Manual readings','Hardware sequence'],horizontal=True,key='reading_source')
    left,right = st.columns([1.2,1],gap='large')
    results = None
    source_inputs = None
    fingerprint = None
    started = time.perf_counter()
    with left:
        if mode == 'Manual readings':
            st.subheader('Hardware readings')
            st.caption('Every slider or number change recalculates automatically. CPU and memory use normalised trace units (0–1), not cores or GB.')
            for key in FIELDS:
                st.session_state['input_' + key] = saved[key]
            st.slider('Cycles per instruction (CPI)',0.0,10.0,step=.01,key='input_cycles_per_instruction',on_change=remember,args=('cycles_per_instruction',))
            st.slider('Memory accesses per instruction',0.0,.1,step=.0001,format='%.4f',key='input_memory_accesses_per_instruction',on_change=remember,args=('memory_accesses_per_instruction',))
            st.markdown('**CPU measurements**')
            a,b = st.columns(2)
            for index,key in enumerate(['average_cpu','maximum_cpu','sample_cpu','requested_cpu']):
                label,maximum,step = FIELDS[key]
                (a if index%2==0 else b).number_input(label,min_value=0.0,max_value=maximum,step=step,format='%.6f',key='input_' + key,on_change=remember,args=(key,))
            st.markdown('**Memory measurements**')
            a,b = st.columns(2)
            for index,key in enumerate(['average_memory','maximum_memory','requested_memory','assigned_memory','page_cache_memory']):
                label,maximum,step = FIELDS[key]
                (a if index%2==0 else b).number_input(label,min_value=0.0,max_value=maximum,step=step,format='%.6f',key='input_' + key,on_change=remember,args=(key,))
            values = dict(saved)
            values['cpu_utilisation'] = values['average_cpu']/(values['requested_cpu']+1e-10)
            values['memory_utilisation'] = values['average_memory']/(values['requested_memory']+1e-10)
            st.caption(f"Derived CPU utilisation: {values['cpu_utilisation']:.3f} · Memory utilisation: {values['memory_utilisation']:.3f}")
            invalid = values['average_cpu'] > values['maximum_cpu'] or values['average_memory'] > values['maximum_memory']
            invalid = invalid or (values['requested_cpu']==0 and values['average_cpu']>0) or (values['requested_memory']==0 and values['average_memory']>0)
            if invalid:
                st.error('Enter consistent readings: peak must be at least average, and positive usage requires a nonzero request. Results are withheld until corrected.')
            else:
                try:
                    results = engine.predict(values)
                    source_inputs = values
                    fingerprint = 'manual:' + json.dumps(values,sort_keys=True)
                except Exception as error:
                    st.error(f'Unable to evaluate readings: {error}')
            st.button('Reset to training medians',on_click=reset_readings)
        else:
            st.subheader('Consecutive hardware observations')
            st.caption('All three models evaluate the same final observation. LSTM uses the final 30 rows. A new upload or endpoint change evaluates immediately.')
            sample = ROOT / 'examples/hardware-sequence.csv'
            st.download_button('Download example CSV',sample.read_bytes(),'hardware-sequence.csv','text/csv')
            uploaded = st.file_uploader('Hardware sequence CSV',type=['csv'])
            use_example = st.checkbox('Use bundled example',value=True)
            content = uploaded.getvalue() if uploaded is not None else (sample.read_bytes() if use_example else None)
            if content is not None:
                try:
                    from io import BytesIO
                    frame = pd.read_csv(BytesIO(content))
                    sequence_engine = load_predictor('smd')
                    sequence_engine.validate_sequence(frame)
                    endpoint = st.slider('Observation to evaluate',30,len(frame),len(frame),key='sequence_endpoint') if len(frame)>30 else 30
                    fingerprint = 'sequence:' + hashlib.sha256(content).hexdigest() + ':' + str(endpoint)
                    if st.session_state.get('sequence_fingerprint') != fingerprint:
                        with st.spinner('Evaluating all three models…'):
                            evaluated = sequence_engine.predict_sequence(frame.iloc[:endpoint])
                        st.session_state['sequence_result'] = evaluated
                        st.session_state['sequence_fingerprint'] = fingerprint
                    results = st.session_state['sequence_result']
                    source_inputs = {'source_sha256':hashlib.sha256(content).hexdigest(),'observations':endpoint,'latest_reading':frame.iloc[endpoint-1].to_dict()}
                    st.caption(f'Endpoint {endpoint} / {len(frame)} · uploaded sequence' if uploaded else f'Endpoint {endpoint} / {len(frame)} · recorded example')
                except Exception as error:
                    st.error(f'Cannot evaluate sequence: {error}')
            else:
                st.info('Upload a sequence or enable the bundled example.')
    with right:
        st.subheader('Current prediction')
        if results is None:
            st.info('Waiting for valid readings. Previous predictions are not shown for invalid inputs.')
        else:
            changed = fingerprint != st.session_state.get('last_fingerprint')
            if changed:
                evaluated_at = datetime.now(ZoneInfo('Africa/Lagos')).isoformat(timespec='seconds')
                previous = st.session_state.get('evaluation',[])
                st.session_state['previous_scores'] = {r['Model']:r['Score'] for r in previous} if st.session_state.get('last_mode')==mode else {}
                st.session_state.update(evaluation=results,evaluated_at=evaluated_at,evaluated_inputs=source_inputs,last_fingerprint=fingerprint,last_mode=mode)
                st.session_state['update_count'] = st.session_state.get('update_count',0)+1
                record = {'Update':st.session_state['update_count'],'Time':evaluated_at,'Source':mode}
                record.update({r['Model']:r['Score'] for r in results})
                st.session_state['history'] = (st.session_state['history'] + [record])[-100:]
                st.session_state['last_latency'] = (time.perf_counter()-started)*1000
            st.caption('Automatically evaluated · ' + st.session_state['evaluated_at'])
            selected = next((r for r in results if r['Model']==model),None)
            if selected is None:
                st.info('LSTM needs at least 30 consecutive hardware observations. Select Hardware sequence to evaluate it.')
            else:
                st.markdown(f'<div class="panel"><div class="eyebrow">{selected["Model"].upper()}</div><h2>{selected["Outcome"]}</h2></div>',unsafe_allow_html=True)
                probability = selected['Failure probability']
                previous_score = st.session_state.get('previous_scores',{}).get(model)
                delta = None if previous_score is None else selected['Score']-previous_score
                if probability is not None:
                    st.metric('Model failure score',f'{probability:.2%}',None if delta is None else f'{delta*100:+.2f} percentage points',delta_color='inverse')
                    st.progress(probability)
                    st.caption('Random Forest class probability for this failure-proxy dataset; not a calibrated production forecast.')
                else:
                    st.metric('Anomaly score',f'{selected["Score"]:.6f}',None if delta is None else f'{delta:+.6f}',delta_color='inverse')
                st.caption(f'Flagged above saved threshold {selected["Threshold"]:.6f}.')
                if changed and delta is not None and abs(delta)<1e-12:
                    st.caption('Inputs were recalculated. The model returned the same score; nearby readings can follow the same tree paths.')
            table = pd.DataFrame(results).drop(columns=['Failure probability'])
            st.dataframe(table,hide_index=True,width='stretch')
            st.caption(f"{len(results)} models evaluated · {st.session_state.get('last_latency',0):.0f} ms · user-entered readings" if mode=='Manual readings' else f'{len(results)} models evaluated · sequence inference')
            if mode=='Manual readings':
                st.caption('Every trained hardware feature is supplied. CPU and memory utilisation are derived from the readings above.')
            history = pd.DataFrame(st.session_state['history'])
            history = history[history.Source==mode]
            chart_columns = [r['Model'] for r in results if r['Model'] in history.columns]
            if len(history)>1:
                st.markdown('**Recent score history**')
                st.caption('Raw model scores use different scales; compare each model with its own prior readings and threshold.')
                for name in chart_columns:
                    st.line_chart(history.set_index('Update')[[name]],height=120,color='#F5D547')
            payload = {'source':mode,'evaluated_at':st.session_state['evaluated_at'],'inputs':source_inputs,'results':results}
            st.download_button('Download evaluation',json.dumps(payload,indent=2),'cloudify-evaluation.json','application/json')
            st.download_button('Download history',pd.DataFrame(st.session_state['history']).to_csv(index=False),'cloudify-history.csv','text/csv')
            st.button('Clear history',on_click=lambda:st.session_state.update(history=[],last_fingerprint=None))
st.divider()
st.caption('CLOUDIFY / Scheduling & Hardware · Predictions update on submitted widget changes · No live provider connection')
