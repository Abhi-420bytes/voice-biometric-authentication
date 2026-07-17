const BASE = '/api'

async function req(path, options = {}) {
  const res = await fetch(BASE + path, options)
  const json = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(json.detail || `HTTP ${res.status}`)
  return json
}

// ── Health ───────────────────────────────────────────────────────
export const getHealth = () => req('/health')

// ── Users ────────────────────────────────────────────────────────
export const getUsers      = ()           => req('/users')
export const getUserStatus = (username)   => req(`/users/${username}`)
export const deleteUser    = (username)   => req(`/users/${username}`, { method: 'DELETE' })

// ── Enroll ───────────────────────────────────────────────────────
export function enrollTextIndependent(username, files) {
  const fd = new FormData()
  fd.append('username', username)
  files.forEach(f => fd.append('files', f))
  return req('/enroll/text-independent', { method: 'POST', body: fd })
}

export function enrollTextDependent(username, files, passphrase) {
  const fd = new FormData()
  fd.append('username', username)
  fd.append('passphrase', passphrase)
  files.forEach(f => fd.append('files', f))
  return req('/enroll/text-dependent', { method: 'POST', body: fd })
}

// ── Authenticate ─────────────────────────────────────────────────
export function authTextIndependent(username, file) {
  const fd = new FormData()
  fd.append('username', username)
  fd.append('file', file)
  return req('/auth/text-independent', { method: 'POST', body: fd })
}

export function authTextDependent(username, file, passphrase) {
  const fd = new FormData()
  fd.append('username', username)
  fd.append('file', file)
  fd.append('passphrase', passphrase)
  return req('/auth/text-dependent', { method: 'POST', body: fd })
}

// ── Spoof ────────────────────────────────────────────────────────
export function detectSpoof(file) {
  const fd = new FormData()
  fd.append('file', file)
  return req('/spoof/detect', { method: 'POST', body: fd })
}
