const BASE = import.meta.env.VITE_API_URL ?? ''

export async function get(path) {
  const response = await fetch(`${BASE}${path}`)
  const data = await response.json()
  if (!response.ok) throw new Error(data.error ?? 'Request failed')
  return data
}
