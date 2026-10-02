// Thin fetch wrapper: JSON in/out. `{ error, code, hint, key, params }` bodies become exceptions that keep code, hint, key and params.
async function request(method, path, { params, body } = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, value);
  }
  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (cause) {
    const error = new Error(cause && cause.message ? cause.message : "Network error");
    error.code = "network";
    throw error;
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: text.slice(0, 300) };
  }
  if (!response.ok) {
    const detail = data && (data.error || data.detail);
    const error = new Error((typeof detail === "string" && detail) || `Error ${response.status}`);
    error.code = data && data.code;
    error.hint = data && data.hint;                     // a message item ({ key, params, text }) or plain text
    error.key = data && data.key;                       // the catalogue key of the error and its parameters: i18n.js words them
    error.params = data && data.params;
    error.status = response.status;
    throw error;
  }
  return data;
}

// Every read and write goes through the same tool handlers the assistant uses.
const call = (name, args) => request("POST", "/api/ui/call", { body: { name, arguments: args || {} } });

export const api = {
  health: () => request("GET", "/api/health"),
  dashboard: () => request("GET", "/api/dashboard"),
  call,
};

// Message for a toast: the backend's error plus its hint, in the language of the UI (`t` is the translator of the app).
export function errorText(error, t) {
  if (!error) return "";
  if (t && t.errorLine) return t.errorLine(error);
  const base = error.message || String(error);
  return typeof error.hint === "string" && error.hint ? `${base} — ${error.hint}` : base;
}
