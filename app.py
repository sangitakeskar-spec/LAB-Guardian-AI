import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title='LAB-GUARDIAN AI | Research Demo', layout='wide', page_icon='🧪')
st.title('🧪 LAB-GUARDIAN AI')
st.subheader('Multidisciplinary Laboratory Specimen Acceptance & Rejection Intelligence System')
st.warning('RESEARCH PROTOTYPE ONLY — Synthetic records, illustrative rules, not clinically validated. Never use for actual specimen disposition. Human authorization and approved test-specific SOPs are required.')
st.caption('Right Specimen. Right Test. Reliable Results. Every Time. | AI suggests → Human verifies → Authorized staff decide')

@st.cache_data
def load_data():
    return pd.read_csv('synthetic_specimens_1000.csv')
df=load_data()

DEMO_CASES={
 'Hematology – clotted CBC': ('Hematology','CBC','Clotted EDTA',False),
 'Biochemistry – interference flag': ('Biochemistry','Chemistry panel','Interference flag',False),
 'Microbiology – identification discrepancy': ('Microbiology','Culture','Identification discrepancy',False),
 'Clinical Pathology – delayed specimen': ('Clinical Pathology','Urine routine','Transport delay',False),
 'Hematology – no apparent issue': ('Hematology','CBC','No issue',False),
}

# Illustrative demonstration logic, NOT an approved laboratory rulebook.
def screen(dept,test,issue,urgent):
    s=issue.lower()
    if 'identification' in s or 'wrong container' in s or 'clotted edta' in s:
        return ('REJECT — proposed', 'Illustrative high-risk nonconformity. Confirm test-specific SOP and notify the requester. Do not process pending authorized decision.', 'DEMO-R01 (not approved)')
    if 'no issue' in s or 'none' == s:
        return ('ACCEPT — proposed', 'No issue recorded in this demonstration. All actual specimen and test requirements must still be verified.', 'DEMO-R00 (not approved)')
    return ('HOLD FOR REVIEW', 'Potential nonconformity requires review against approved test-specific limits, specimen suitability, urgency and clinical circumstances.', 'DEMO-R02 (not approved)')

t1,t2,t3,t4=st.tabs(['🔬 Specimen checker','📊 EDA dashboard','🤖 Model evaluation','📋 Quality indicators & CAPA'])
with t1:
    st.markdown('### Test a specimen scenario')
    demo=st.selectbox('Load a demo specimen', ['Custom entry']+list(DEMO_CASES))
    defaults=DEMO_CASES.get(demo,('Hematology','CBC','No issue',False))
    with st.form('checker'):
        dept=st.selectbox('Department',sorted(df.department.unique()),index=sorted(df.department.unique()).index(defaults[0]) if defaults[0] in df.department.unique() else 0)
        test=st.text_input('Requested test',value=defaults[1])
        issue=st.selectbox('Observed issue',sorted(set(df.issue.astype(str)).union({'No issue','Clotted EDTA','Identification discrepancy','Wrong container','Transport delay','Interference flag'})),index=sorted(set(df.issue.astype(str)).union({'No issue','Clotted EDTA','Identification discrepancy','Wrong container','Transport delay','Interference flag'})).index(defaults[2]) if defaults[2] in set(df.issue.astype(str)).union({'No issue','Clotted EDTA','Identification discrepancy','Wrong container','Transport delay','Interference flag'}) else 0)
        urgent=st.checkbox('Urgent or irreplaceable specimen',value=defaults[3])
        run=st.form_submit_button('Analyze specimen')
    if run:
        decision,reason,rule=screen(dept,test,issue,urgent)
        if urgent and not decision.startswith('ACCEPT'):
            decision='HOLD FOR URGENT AUTHORIZED REVIEW'
            reason='Urgent or irreplaceable specimen: escalate immediately. Do not permit an automatic rejection or exception.'
        st.info(f'**Proposed disposition:** {decision}')
        st.write('**Reason / action:**',reason)
        st.write('**Rule reference:**',rule)
        st.write('**Human reviewer:** Not completed | **Final authorized decision:** Pending')
        st.caption('This prototype does not measure tube fill, transport conditions, analyte-specific interference or SOP thresholds. Never enter patient identifiers.')
with t2:
    st.markdown('### Exploratory data analysis — 1,000 synthetic specimens')
    depts=st.multiselect('Filter departments',sorted(df.department.unique()),default=sorted(df.department.unique()))
    view=df[df.department.isin(depts)]
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Specimens',len(view));c2.metric('Accepted',int((view.decision=='ACCEPT').sum()));c3.metric('Rejected',int((view.decision=='REJECT').sum()));c4.metric('Held for review',int((view.decision=='HOLD FOR REVIEW').sum()))
    if len(view):
        counts = view.groupby(['department', 'decision']).size().reset_index(name='Count')
st.plotly_chart(px.bar(counts, x='department', y='Count', color='decision', barmode='group', title='Decision by Department'), use_container_width=True)
        st.plotly_chart(px.bar(view.issue.value_counts().head(12).rename_axis('Issue').reset_index(name='Count'),x='Issue',y='Count',title='Top recorded specimen issues'),use_container_width=True)
        month=view.groupby(['month','decision']).size().reset_index(name='Count')
        st.plotly_chart(px.line(month,x='month',y='Count',color='decision',markers=True,title='Synthetic monthly trends'),use_container_width=True)
    st.download_button('Download filtered synthetic data (CSV)',view.to_csv(index=False).encode(),file_name='lab_guardian_filtered.csv',mime='text/csv')
    st.dataframe(view.head(100),use_container_width=True)
with t3:
    st.markdown('### Reproducible 80:20 machine-learning evaluation')
    st.info('This classifier predicts synthetic decision labels. Scores do NOT establish expert agreement, clinical validity or safety.')
    X=df[['department','test','issue','urgent','collection_site','month']]
    y=df['decision']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),['department','test','issue','collection_site']),('numeric','passthrough',['urgent','month'])])
    model=Pipeline([('prep',pre),('classifier',RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced'))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    a,b,c=st.columns(3);a.metric('Training records',len(X_train));b.metric('Testing records',len(X_test));c.metric('Test accuracy',f'{accuracy_score(y_test,pred):.1%}')
    st.metric('Balanced accuracy',f'{balanced_accuracy_score(y_test,pred):.1%}')
    labels=sorted(y.unique())
    cm=confusion_matrix(y_test,pred,labels=labels)
    st.plotly_chart(px.imshow(cm,x=labels,y=labels,text_auto=True,labels={'x':'Predicted','y':'Actual'},title='Confusion matrix — synthetic holdout'),use_container_width=True)
    st.dataframe(pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T,use_container_width=True)
    st.caption('Random split may share similar synthetic scenario templates across training and test sets. A future expert-labelled, independent dataset is required.')
with t4:
    st.markdown('### Quality indicators and corrective-action planning')
    n=len(df);reject=(df.decision=='REJECT').sum()
    st.metric('Synthetic rejection rate',f'{reject/n:.1%}')
    st.write('**Nine workstreams:** Identification discrepancy; wrong container; clotted EDTA; inadequate volume; interference; transport; microbiology specimen quality; incomplete requests; audit documentation.')
    st.write('**KPI denominator caution:** Department- and test-specific indicators need their own eligible denominators. This demonstration only reports fields actually present in the dataset.')
    st.dataframe(pd.crosstab(df.department,df.decision,margins=True),use_container_width=True)
    st.markdown('**Illustrative CAPA register (not measured outcomes)**')
    st.dataframe(pd.DataFrame([
        {'Problem':'Clotted EDTA','Potential root cause':'Collection or mixing technique','Suggested action':'Review phlebotomy training and collection SOP','Effectiveness measure':'Clotted EDTA rate over time'},
        {'Problem':'Identification discrepancy','Potential root cause':'Bedside labelling failure','Suggested action':'Audit two-identifier checks','Effectiveness measure':'Identification events per 1,000 specimens'},
        {'Problem':'Transport delay','Potential root cause':'Courier workflow','Suggested action':'Map collection-to-receipt process','Effectiveness measure':'Transport nonconformity rate'}
    ]),hide_index=True,use_container_width=True)
    st.error('No actual reviewer agreement, false rejection rate, notification time, recollection completion, or CAPA effectiveness can be established from this synthetic dataset.')
