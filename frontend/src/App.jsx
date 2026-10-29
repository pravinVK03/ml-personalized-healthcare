import { useEffect, useMemo, useState } from 'react'
import { get } from './api.js'

const MODELS = [
  ['persistence', 'Persistence baseline'],
  ['ridge', 'Ridge regression'],
  ['random_forest', 'Random forest'],
  ['gradient_boosting', 'Histogram gradient boosting'],
]

const ENCOUNTER_MODELS = [
  ['logistic_regression', 'Logistic regression'],
  ['random_forest', 'Random forest'],
  ['histogram_gradient_boosting', 'Histogram gradient boosting'],
]

// Keep the presentation map beside the integrated UI.  Each dataset still has
// its own loading and model module; only the results are brought together here.
const CODE_AREAS = {
  shanghai_t2d: {
    source: 'src/shanghai_t2d/',
    approach: 'CGM history → future glucose forecast',
    evaluation: 'Patient-grouped cross-validation',
  },
  us_diabetes_130: {
    source: 'src/us_diabetes_130.py',
    approach: 'Discharge encounter summary → 30-day readmission',
    evaluation: 'Patient-held-out test split',
  },
  mimic_iv_demo: {
    source: 'src/mimic_iv_demo.py',
    approach: 'Admission summary → in-hospital mortality',
    evaluation: 'Patient-held-out demo test split',
  },
}

const fmt = (value, digits = 1) => Number(value).toFixed(digits)
const date = (value) => new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })

function Metric({ label, value, unit = '' }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</article>
}

function Notice({ children, kind = 'info' }) {
  return <div className={`notice ${kind}`}>{children}</div>
}

function DatasetCard({ item, selected, onSelect }) {
  const ready = item.status === 'ready'
  return <button className={`dataset-card ${selected ? 'selected' : ''}`} onClick={onSelect} type="button" aria-pressed={selected}>
    <span className={`status-dot ${ready ? 'ready' : ''}`} aria-hidden="true" />
    <span className="dataset-card-body"><strong>{item.name}</strong><small>{item.target}</small></span>
    <span className={`status-label ${ready ? 'ready' : ''}`}>{ready ? 'Results ready' : 'Data needed'}</span>
  </button>
}

function StudyMap({ datasetId, dataset }) {
  const code = CODE_AREAS[datasetId]
  if (!code || !dataset) return null
  return <section className="panel study-map">
    <div><p className="section-label">CODE AND EVALUATION</p><h2>How this study is organised</h2><p>Dataset processing and model training stay separate. The dashboard only presents their saved, held-out results.</p></div>
    <div className="map-grid">
      <article><span>Dataset module</span><strong><code>{code.source}</code></strong></article>
      <article><span>Prediction flow</span><strong>{code.approach}</strong></article>
      <article><span>Evaluation design</span><strong>{code.evaluation}</strong></article>
      <article><span>Shared support</span><strong><code>{datasetId === 'shanghai_t2d' ? 'src/shanghai_t2d/evaluation.py' : 'src/encounter_common.py'}</code></strong></article>
    </div>
  </section>
}

function StudyPurpose({ dataset }) {
  return <section className="panel purpose-panel">
    <p className="section-label">STEP 2 — PURPOSE, INPUTS, AND FORECAST</p>
    <h2>What is forecasted, using which inputs?</h2>
    <p className="purpose-copy">{dataset.purpose}</p>
    <div className="forecast-flow" aria-label="Prediction flow">
      <article><span>1. Input boundary</span><strong>{dataset.input_boundary}</strong></article>
      <i aria-hidden="true">→</i>
      <article><span>2. Patient state</span><strong>{dataset.state_definition}</strong></article>
      <i aria-hidden="true">→</i>
      <article className="forecast-result"><span>3. Forecasted result</span><strong>{dataset.target}</strong></article>
    </div>
    <div className="feature-area"><div><h3>Features used by the model</h3><p>Feature groups are kept specific to this dataset and prediction question.</p></div><div className="feature-groups">{dataset.feature_groups?.map(group => <article key={group.name}><strong>{group.name}</strong><ul>{group.items.map(item => <li key={item}>{item}</li>)}</ul></article>)}</div></div>
  </section>
}

function LineChart({ history, forecast }) {
  const values = [...history.map(point => point.cgm), ...forecast.map(point => point.forecast_cgm), ...forecast.map(point => point.measured_later_cgm)]
  if (!values.length) return null
  const min = Math.min(...values) - 10
  const max = Math.max(...values) + 10
  const width = 720, height = 260, pad = 32
  const all = [
    ...history.map(point => ({ time: new Date(point.time).getTime(), value: point.cgm, type: 'history' })),
    ...forecast.map(point => ({ time: new Date(point.time).getTime(), value: point.forecast_cgm, type: 'forecast' })),
  ]
  const measured = forecast.map(point => ({ time: new Date(point.time).getTime(), value: point.measured_later_cgm }))
  const times = [...all, ...measured].map(point => point.time)
  const start = Math.min(...times), end = Math.max(...times)
  const x = time => pad + ((time - start) / Math.max(end - start, 1)) * (width - 2 * pad)
  const y = value => height - pad - ((value - min) / Math.max(max - min, 1)) * (height - 2 * pad)
  const path = points => points.map((point, index) => `${index ? 'L' : 'M'} ${x(point.time)} ${y(point.value)}`).join(' ')
  return <div className="chart-wrap">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recorded glucose history and saved forecast">
      <line x1={pad} x2={width - pad} y1={y(min)} y2={y(min)} className="axis" />
      <line x1={pad} x2={pad} y1={pad} y2={height - pad} className="axis" />
      <text x={4} y={pad} className="axis-text">{fmt(max)} mg/dL</text>
      <text x={4} y={height - pad} className="axis-text">{fmt(min)} mg/dL</text>
      <path d={path(history.map(point => ({ time: new Date(point.time).getTime(), value: point.cgm })))} className="history-line" />
      <path d={path(forecast.map(point => ({ time: new Date(point.time).getTime(), value: point.forecast_cgm })))} className="forecast-line" />
      {measured.map((point, index) => <circle key={index} cx={x(point.time)} cy={y(point.value)} r="4" className="measured-dot" />)}
    </svg>
    <div className="legend"><span><i className="history-key" />Recorded CGM</span><span><i className="forecast-key" />Forecast CGM</span><span><i className="measured-key" />Measured later CGM</span></div>
  </div>
}

function OutcomeGraph({ state, target }) {
  if (!state) return null
  const percent = Math.round(state.predicted_probability * 100)
  return <section className="panel outcome-graph"><p className="section-label">PATIENT STATE TO OUTCOME</p><h2>Outcome path for this held-out encounter</h2><p>{target}. This is a single encounter prediction, not a disease trajectory: this source has no repeated within-person time series.</p>
    <div className="outcome-track" role="img" aria-label={`Predicted probability ${percent} percent`}><span className="outcome-fill" style={{ width: `${percent}%` }} /><b style={{ left: `${percent}%` }}>{percent}%</b></div>
    <div className="outcome-labels"><span>Lower predicted probability</span><span>Higher predicted probability</span></div>
  </section>
}

function Accuracy({ rows }) {
  const modelOrder = Object.fromEntries(MODELS.map(([id], index) => [id, index]))
  const sorted = [...rows].sort((a, b) => a.horizon_minutes - b.horizon_minutes || modelOrder[a.model] - modelOrder[b.model])
  const hourAhead = sorted.filter(row => row.horizon_minutes === 60)
  const largestError = Math.max(...hourAhead.map(row => row.estimate), 1)
  return <section className="panel"><p className="section-label">STEP 5 — HELD-OUT EVALUATION</p><h2>Overall held-out accuracy</h2><p>MAE is the average absolute forecast error in mg/dL. Lower is better. Every participant was held out of model training.</p>
    <div className="performance-grid">{hourAhead.map(row => <article key={row.model}><strong>{row.model_label}</strong><span>60-minute MAE: {fmt(row.estimate, 2)} mg/dL</span><div className="performance-track" aria-label={`${row.model_label}: ${fmt(row.estimate, 2)} MAE, lower is better`}><i className="error-bar" style={{ width: `${(row.estimate / largestError) * 100}%` }} /></div><small>Shorter bar = lower error</small></article>)}</div>
    <div className="table-scroll"><table><thead><tr><th>Model</th><th>Minutes ahead</th><th>MAE</th><th>95% interval</th><th>Participants</th></tr></thead>
      <tbody>{sorted.map(row => <tr key={`${row.model}-${row.horizon_minutes}`}><td>{row.model_label}</td><td>{row.horizon_minutes}</td><td>{fmt(row.estimate, 2)}</td><td>{fmt(row.ci_2_5, 2)} to {fmt(row.ci_97_5, 2)}</td><td>{row.n_subjects}</td></tr>)}</tbody>
    </table></div></section>
}

function EncounterAccuracy({ rows }) {
  const smallSample = rows[0]?.n_test_patients < 30
  return <section className="panel"><p className="section-label">STEP 5 — HELD-OUT EVALUATION</p><h2>Overall held-out accuracy</h2><p>Each model used a patient-held-out evaluation split. Balanced accuracy gives equal importance to both outcomes; ROC-AUC measures ranking ability. Higher is better.</p>
    {smallSample && <p className="small-sample">Feasibility warning: this demo evaluation has only {rows[0]?.n_test_patients} held-out patients. The displayed values are not clinically generalizable.</p>}
    <div className="performance-grid">{rows.map(row => <article key={row.model}><strong>{row.model_label}</strong><span>Balanced accuracy: {fmt(row.balanced_accuracy, 3)}</span><div className="performance-track"><i style={{ width: `${row.balanced_accuracy * 100}%` }} /></div><span>ROC-AUC: {fmt(row.roc_auc, 3)}</span><div className="performance-track"><i className="roc-bar" style={{ width: `${row.roc_auc * 100}%` }} /></div></article>)}</div>
    <div className="table-scroll"><table><thead><tr><th>Model</th><th>Balanced accuracy</th><th>ROC-AUC</th><th>Average precision</th><th>Test records</th><th>Positive outcome rate</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.model}><td>{row.model_label}</td><td>{fmt(row.balanced_accuracy, 3)}</td><td>{fmt(row.roc_auc, 3)}</td><td>{fmt(row.average_precision, 3)}</td><td>{row.n_test}</td><td>{fmt(row.positive_rate_test * 100, 1)}%</td></tr>)}</tbody>
    </table></div></section>
}

function EncounterRecord({ state, target }) {
  if (!state) return null
  const outcome = state.observed_outcome ? 'Yes' : 'No'
  const predicted = state.predicted_class ? 'Yes' : 'No'
  return <>
    <section className="panel"><p className="section-label">STEP 4 — SELECTED PATIENT STATE</p><h2>Patient state at the prediction time</h2><p>{target}. These are the documented fields supplied to the model for this held-out record.</p>
      <div className="table-scroll"><table><thead><tr><th>Field</th><th>Recorded value</th></tr></thead><tbody>{Object.entries(state.state).map(([key, value]) => <tr key={key}><td>{key.replaceAll('_', ' ')}</td><td>{String(value ?? 'Not recorded')}</td></tr>)}</tbody></table></div>
    </section>
    <OutcomeGraph state={state} target={target} />
    <section className="panel"><p className="section-label">SELECTED RECORD RESULT</p><h2>Held-out outcome result</h2><p>The observed outcome is shown only for retrospective evaluation; it was not available to the model when predicting.</p><div className="metrics">
      <Metric label="Predicted probability" value={`${fmt(state.predicted_probability * 100, 1)}%`} />
      <Metric label="Predicted outcome" value={predicted} />
      <Metric label="Observed outcome" value={outcome} />
    </div></section>
    <section className="panel"><h2>Confusion matrix</h2><p>Counts for the selected model across all held-out records.</p><div className="table-scroll"><table><thead><tr><th>Observed outcome</th><th>Predicted outcome</th><th>Count</th></tr></thead><tbody>{state.confusion.map(row => <tr key={`${row.actual}-${row.predicted}`}><td>{row.actual ? 'Yes' : 'No'}</td><td>{row.predicted ? 'Yes' : 'No'}</td><td>{row.count}</td></tr>)}</tbody></table></div></section>
  </>
}

function DatasetScope({ dataset }) {
  return <section className="panel scope"><p className="section-label">STUDY OVERVIEW</p><h2>{dataset.name}</h2><dl>
    <dt>Care setting</dt><dd>{dataset.setting}</dd>
    <dt>Cohort definition</dt><dd>{dataset.cohort_definition}</dd>
    <dt>Models compared</dt><dd>{dataset.models.join(', ')}</dd>
    <dt>Interpret carefully</dt><dd>{dataset.limitations}</dd>
  </dl></section>
}

export default function App() {
  const [datasets, setDatasets] = useState([])
  const [datasetId, setDatasetId] = useState('shanghai_t2d')
  const [dataset, setDataset] = useState(null)
  const [participants, setParticipants] = useState([])
  const [participant, setParticipant] = useState('')
  const [episodes, setEpisodes] = useState([])
  const [episode, setEpisode] = useState('')
  const [model, setModel] = useState('gradient_boosting')
  const [state, setState] = useState(null)
  const [records, setRecords] = useState([])
  const [recordsForDataset, setRecordsForDataset] = useState('')
  const [record, setRecord] = useState('')
  const [recordState, setRecordState] = useState(null)
  const [origin, setOrigin] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    get('/api/datasets').then(data => { if (!cancelled) setDatasets(data.datasets) }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [])
  useEffect(() => {
    let cancelled = false
    setDataset(null); setParticipants([]); setParticipant(''); setEpisodes([]); setEpisode(''); setState(null); setRecords([]); setRecordsForDataset(''); setRecord(''); setRecordState(null); setOrigin(''); setModel(datasetId === 'shanghai_t2d' ? 'gradient_boosting' : 'logistic_regression'); setError('')
    get(`/api/datasets/${datasetId}`).then(data => { if (!cancelled) setDataset(data) }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [datasetId])
  useEffect(() => {
    if (dataset?.status !== 'ready' || datasetId !== 'shanghai_t2d') return
    let cancelled = false
    get(`/api/datasets/${datasetId}/participants`).then(data => { if (!cancelled) { setParticipants(data.participants); setParticipant(data.participants[0] ?? '') } }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [dataset, datasetId])
  useEffect(() => {
    if (dataset?.status !== 'ready' || datasetId === 'shanghai_t2d') return
    let cancelled = false
    setRecords([]); setRecordsForDataset(''); setRecord(''); setRecordState(null)
    get(`/api/datasets/${datasetId}/records?model=${model}`).then(data => { if (!cancelled) { setRecords(data.records); setRecordsForDataset(datasetId); setRecord(data.records[0]?.record_id ?? '') } }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [dataset, datasetId, model])
  useEffect(() => {
    if (datasetId !== 'shanghai_t2d' || !participant) return
    let cancelled = false
    get(`/api/datasets/${datasetId}/participants/${participant}/episodes`).then(data => { if (!cancelled) { setEpisodes(data.episodes); setEpisode(data.episodes[0] ?? '') } }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [datasetId, participant])
  useEffect(() => {
    if (datasetId !== 'shanghai_t2d' || !participant || !episode) return
    let cancelled = false
    const params = new URLSearchParams({ episode, model }); if (origin) params.set('origin', origin)
    get(`/api/datasets/${datasetId}/participants/${participant}/state?${params}`).then(data => { if (!cancelled) { setState(data); setOrigin(data.selected_origin) } }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [datasetId, participant, episode, model, origin])
  useEffect(() => {
    if (!record || datasetId === 'shanghai_t2d' || recordsForDataset !== datasetId) return
    let cancelled = false
    get(`/api/datasets/${datasetId}/records/${record}?model=${model}`).then(data => { if (!cancelled) setRecordState(data) }).catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [datasetId, record, model, recordsForDataset])

  const selectedDataset = useMemo(() => datasets.find(item => item.id === datasetId), [datasets, datasetId])
  const best60 = dataset?.evaluation?.accuracy?.filter(row => row.horizon_minutes === 60).sort((a, b) => a.estimate - b.estimate)[0]
  const unavailable = dataset && dataset.status !== 'ready'
  const isShanghai = datasetId === 'shanghai_t2d'
  const availableModels = isShanghai ? MODELS : ENCOUNTER_MODELS

  return <main>
    <header><div><p className="eyebrow">RETROSPECTIVE RESEARCH DASHBOARD</p><h1>T2D-Focused Patient State<br />and Future Outcome Modelling</h1><p className="subtitle">Three separate datasets are shown with their cohort definitions. Results are retrospective research outputs, not clinical advice.</p></div><div className="header-stats"><span>3 datasets</span><span>Standard scikit-learn models</span><span>Held-out evaluation</span></div></header>
    {error && <Notice kind="error">{error}</Notice>}
    <section className="dataset-picker"><div><p className="section-label">STEP 1</p><h2>Choose a study dataset</h2><p>Each dataset has its own target, features, and evaluation. Select one to inspect its patient-state example and model results.</p></div><div className="dataset-cards">{datasets.map(item => <DatasetCard key={item.id} item={item} selected={item.id === datasetId} onSelect={() => setDatasetId(item.id)} />)}</div></section>
    {selectedDataset && <DatasetScope dataset={dataset ?? selectedDataset} />}
    {unavailable && <Notice>This dataset is intentionally disabled: {dataset.message} Expected location: <code>{dataset.required_path}</code>. No model, forecast, or accuracy result is shown until compatible data are audited and evaluated.</Notice>}
    {dataset?.status === 'ready' && <>
      <StudyPurpose dataset={dataset} />
      <StudyMap datasetId={datasetId} dataset={dataset} />
      {isShanghai && best60 && <Notice><strong>Best 60-minute result:</strong> {best60.model_label}, average error {fmt(best60.estimate, 2)} mg/dL. This is a held-out retrospective result.</Notice>}
      {isShanghai ? <><section className="selection panel"><p className="section-label">STEP 3</p><h2>Choose a held-out CGM record</h2><div className="controls">
        <label>Study participant<select value={participant} onChange={event => setParticipant(event.target.value)}>{participants.map(item => <option key={item}>{item}</option>)}</select></label>
        <label>CGM recording<select value={episode} onChange={event => setEpisode(event.target.value)}>{episodes.map(item => <option key={item}>{item}</option>)}</select></label>
        <label>Model<select value={model} onChange={event => setModel(event.target.value)}>{availableModels.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
        <label>Prediction time<select value={origin} onChange={event => setOrigin(event.target.value)}>{state?.origins?.map(item => <option key={item} value={item}>{date(item)}</option>)}</select></label>
      </div></section>
      {state && <>
        <section className="panel"><p className="section-label">STEP 4 — PATIENT STATE</p><h2>Patient state at the prediction time</h2><p>These are past measurements already available before the chosen prediction time.</p><div className="metrics">
          <Metric label="Last CGM measurement" value={fmt(state.state.last_cgm_mg_dl)} unit="mg/dL" />
          <Metric label="Change over previous hour" value={`${state.state.one_hour_change_mg_dl >= 0 ? '+' : ''}${fmt(state.state.one_hour_change_mg_dl)}`} unit="mg/dL" />
          <Metric label="Mean over last six hours" value={fmt(state.state.six_hour_mean_mg_dl)} unit="mg/dL" />
          <Metric label="Variation over last six hours" value={fmt(state.state.six_hour_variation_mg_dl)} unit="mg/dL" />
          <Metric label="Direction over previous hour" value={state.state.direction} />
        </div></section>
        <section className="panel"><p className="section-label">DISEASE TRAJECTORY</p><h2>Recorded glucose path and forecast</h2><p>{dataset.trajectory_note} The later CGM measurement is visible only because this is retrospective evaluation; it was not available at the prediction time.</p><LineChart history={state.history} forecast={state.forecast} />
          <div className="table-scroll"><table><thead><tr><th>Minutes ahead</th><th>Later measurement time</th><th>Forecast CGM</th><th>Measured later CGM</th><th>Absolute error</th></tr></thead><tbody>{state.forecast.map(row => <tr key={row.minutes_ahead}><td>{row.minutes_ahead}</td><td>{date(row.time)}</td><td>{fmt(row.forecast_cgm)}</td><td>{fmt(row.measured_later_cgm)}</td><td>{fmt(row.absolute_error)}</td></tr>)}</tbody></table></div>
        </section>
      </>}
      {dataset.evaluation && <Accuracy rows={dataset.evaluation.accuracy} />}</> : <>
        <section className="selection panel"><p className="section-label">STEP 3</p><h2>Choose a held-out encounter</h2><div className="controls">
          <label>Model<select value={model} onChange={event => setModel(event.target.value)}>{availableModels.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
          <label>Held-out record<select value={record} onChange={event => setRecord(event.target.value)}>{records.map(item => <option key={item.record_id} value={item.record_id}>Patient {item.patient_id} — record {item.record_id}</option>)}</select></label>
        </div></section>
        <EncounterRecord state={recordState} target={dataset.target} />
        {dataset.evaluation && <EncounterAccuracy rows={dataset.evaluation.accuracy} />}
      </>}
    </>}
    <footer>Patient state means a documented summary of available observations. It does not represent a biological twin, provide treatment advice, or demonstrate clinical benefit.</footer>
  </main>
}
