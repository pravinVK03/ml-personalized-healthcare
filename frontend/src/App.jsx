import { useEffect, useMemo, useState } from 'react'
import { get } from './api.js'

const MODELS = [
  ['persistence', 'Persistence baseline'],
  ['ridge', 'Ridge regression'],
  ['random_forest', 'Random forest'],
  ['gradient_boosting', 'Histogram gradient boosting'],
]

const fmt = (value, digits = 1) => Number(value).toFixed(digits)
const date = (value) => new Date(value).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })

function Metric({ label, value, unit = '' }) {
  return <article className="metric"><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</article>
}

function Notice({ children, kind = 'info' }) {
  return <div className={`notice ${kind}`}>{children}</div>
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

function Accuracy({ rows }) {
  const modelOrder = Object.fromEntries(MODELS.map(([id], index) => [id, index]))
  const sorted = [...rows].sort((a, b) => a.horizon_minutes - b.horizon_minutes || modelOrder[a.model] - modelOrder[b.model])
  return <section className="panel"><h2>Overall held-out accuracy</h2><p>MAE is the average absolute forecast error in mg/dL. Lower is better. Every participant was held out of model training.</p>
    <div className="table-scroll"><table><thead><tr><th>Model</th><th>Minutes ahead</th><th>MAE</th><th>95% interval</th><th>Participants</th></tr></thead>
      <tbody>{sorted.map(row => <tr key={`${row.model}-${row.horizon_minutes}`}><td>{row.model_label}</td><td>{row.horizon_minutes}</td><td>{fmt(row.estimate, 2)}</td><td>{fmt(row.ci_2_5, 2)} to {fmt(row.ci_97_5, 2)}</td><td>{row.n_subjects}</td></tr>)}</tbody>
    </table></div></section>
}

function DatasetScope({ dataset }) {
  return <section className="panel scope"><h2>{dataset.name}</h2><dl>
    <dt>Clinical setting</dt><dd>{dataset.setting}</dd>
    <dt>Supported prediction target</dt><dd>{dataset.target}</dd>
    <dt>Patient state representation</dt><dd>{dataset.state_definition}</dd>
    <dt>Model family</dt><dd>{dataset.models.join(', ')}</dd>
    <dt>Important limitation</dt><dd>{dataset.limitations}</dd>
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
  const [origin, setOrigin] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { get('/api/datasets').then(data => setDatasets(data.datasets)).catch(err => setError(err.message)) }, [])
  useEffect(() => {
    setDataset(null); setParticipants([]); setParticipant(''); setEpisodes([]); setEpisode(''); setState(null); setOrigin(''); setError('')
    get(`/api/datasets/${datasetId}`).then(data => setDataset(data)).catch(err => setError(err.message))
  }, [datasetId])
  useEffect(() => {
    if (dataset?.status !== 'ready' || datasetId !== 'shanghai_t2d') return
    get(`/api/datasets/${datasetId}/participants`).then(data => { setParticipants(data.participants); setParticipant(data.participants[0] ?? '') }).catch(err => setError(err.message))
  }, [dataset, datasetId])
  useEffect(() => {
    if (!participant) return
    get(`/api/datasets/${datasetId}/participants/${participant}/episodes`).then(data => { setEpisodes(data.episodes); setEpisode(data.episodes[0] ?? '') }).catch(err => setError(err.message))
  }, [datasetId, participant])
  useEffect(() => {
    if (!episode) return
    const params = new URLSearchParams({ episode, model }); if (origin) params.set('origin', origin)
    get(`/api/datasets/${datasetId}/participants/${participant}/state?${params}`).then(data => { setState(data); setOrigin(data.selected_origin) }).catch(err => setError(err.message))
  }, [datasetId, participant, episode, model, origin])

  const selectedDataset = useMemo(() => datasets.find(item => item.id === datasetId), [datasets, datasetId])
  const best60 = dataset?.evaluation?.accuracy?.filter(row => row.horizon_minutes === 60).sort((a, b) => a.estimate - b.estimate)[0]
  const unavailable = dataset && dataset.status !== 'ready'

  return <main>
    <header><div><p className="eyebrow">RETROSPECTIVE RESEARCH ONLY</p><h1>Patient State and Forecasting</h1><p className="subtitle">A transparent interface for comparing dataset-specific patient-state models. It is not a physiological simulation, live monitoring system, or clinical decision tool.</p></div></header>
    {error && <Notice kind="error">{error}</Notice>}
    <section className="dataset-picker"><label htmlFor="dataset">Dataset</label><select id="dataset" value={datasetId} onChange={event => setDatasetId(event.target.value)}>{datasets.map(item => <option key={item.id} value={item.id}>{item.name} — {item.status === 'ready' ? 'ready' : 'not configured'}</option>)}</select></section>
    {selectedDataset && <DatasetScope dataset={dataset ?? selectedDataset} />}
    {unavailable && <Notice>This dataset is intentionally disabled: {dataset.message} Expected location: <code>{dataset.required_path}</code>. No model, forecast, or accuracy result is shown until compatible data are audited and evaluated.</Notice>}
    {dataset?.status === 'ready' && <>
      {best60 && <Notice><strong>Current best 60-minute Shanghai result:</strong> {best60.model_label}, MAE {fmt(best60.estimate, 2)} mg/dL. This is a held-out retrospective result.</Notice>}
      <section className="selection panel"><h2>Select a held-out study record</h2><div className="controls">
        <label>Study participant<select value={participant} onChange={event => setParticipant(event.target.value)}>{participants.map(item => <option key={item}>{item}</option>)}</select></label>
        <label>CGM recording<select value={episode} onChange={event => setEpisode(event.target.value)}>{episodes.map(item => <option key={item}>{item}</option>)}</select></label>
        <label>Model<select value={model} onChange={event => setModel(event.target.value)}>{MODELS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
        <label>Prediction time<select value={origin} onChange={event => setOrigin(event.target.value)}>{state?.origins?.map(item => <option key={item} value={item}>{date(item)}</option>)}</select></label>
      </div></section>
      {state && <>
        <section className="panel"><h2>Recorded CGM state</h2><p>These are past measurements already available before the chosen prediction time.</p><div className="metrics">
          <Metric label="Last CGM measurement" value={fmt(state.state.last_cgm_mg_dl)} unit="mg/dL" />
          <Metric label="Change over previous hour" value={`${state.state.one_hour_change_mg_dl >= 0 ? '+' : ''}${fmt(state.state.one_hour_change_mg_dl)}`} unit="mg/dL" />
          <Metric label="Mean over last six hours" value={fmt(state.state.six_hour_mean_mg_dl)} unit="mg/dL" />
          <Metric label="Variation over last six hours" value={fmt(state.state.six_hour_variation_mg_dl)} unit="mg/dL" />
          <Metric label="Direction over previous hour" value={state.state.direction} />
        </div></section>
        <section className="panel"><h2>Held-out forecast evaluation</h2><p>The later CGM measurement is visible here only because this is retrospective evaluation. It was not available at the prediction time.</p><LineChart history={state.history} forecast={state.forecast} />
          <div className="table-scroll"><table><thead><tr><th>Minutes ahead</th><th>Later measurement time</th><th>Forecast CGM</th><th>Measured later CGM</th><th>Absolute error</th></tr></thead><tbody>{state.forecast.map(row => <tr key={row.minutes_ahead}><td>{row.minutes_ahead}</td><td>{date(row.time)}</td><td>{fmt(row.forecast_cgm)}</td><td>{fmt(row.measured_later_cgm)}</td><td>{fmt(row.absolute_error)}</td></tr>)}</tbody></table></div>
        </section>
      </>}
      {dataset.evaluation && <Accuracy rows={dataset.evaluation.accuracy} />}
    </>}
    <footer>Patient state means a documented summary of available observations. It does not represent a biological twin, provide treatment advice, or demonstrate clinical benefit.</footer>
  </main>
}
