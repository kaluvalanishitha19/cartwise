import { useState, useEffect } from 'react'

type Message = {
  role: 'user' | 'agent'
  text: string
}

type PendingAction = {
  action: string
  order_id: number
} | null

type Escalation = {
  id: number
  order_id: number | null
  reason: string
  customer_message: string
  agent_note: string
  status: string
  created_at: string
}

function ChatView() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [pendingAction, setPendingAction] = useState<PendingAction>(null)

  async function sendMessage() {
    if (!input.trim()) return
    const userMessage: Message = { role: 'user', text: input }
    setMessages((prev) => [...prev, userMessage])
    setInput('')

    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input, pending_action: pendingAction }),
    })
    const data = await response.json()

    const agentMessage: Message = { role: 'agent', text: data.reply }
    setMessages((prev) => [...prev, agentMessage])
    setPendingAction(data.pending_action ?? null)
  }

  return (
    <div>
      <div style={{ border: '1px solid #ccc', borderRadius: 8, padding: 16, minHeight: 300 }}>
        {messages.map((m, i) => (
          <div key={i} style={{ textAlign: m.role === 'user' ? 'right' : 'left', margin: '8px 0' }}>
            <span style={{
              display: 'inline-block',
              padding: '8px 12px',
              borderRadius: 12,
              background: m.role === 'user' ? '#0f6e56' : '#eee',
              color: m.role === 'user' ? 'white' : 'black',
            }}>
              {m.text}
            </span>
          </div>
        ))}
        {pendingAction && (
          <div style={{ fontSize: 12, color: '#b45309', marginTop: 8 }}>
            Waiting for your confirmation on order #{pendingAction.order_id}...
          </div>
        )}
      </div>

      <div style={{ display: 'flex', marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about your order or our policies..."
          style={{ flex: 1, padding: 8 }}
        />
        <button onClick={sendMessage} style={{ marginLeft: 8, padding: '8px 16px' }}>
          Send
        </button>
      </div>
    </div>
  )
}

function DashboardView() {
  const [escalations, setEscalations] = useState<Escalation[]>([])
  const [loading, setLoading] = useState(true)

  async function loadEscalations() {
    setLoading(true)
    const response = await fetch('http://localhost:8000/escalations')
    const data = await response.json()
    setEscalations(data)
    setLoading(false)
  }

  useEffect(() => {
    loadEscalations()
  }, [])

  async function resolve(id: number) {
    await fetch(`http://localhost:8000/escalations/${id}/resolve`, { method: 'POST' })
    loadEscalations()
  }

  const reasonLabels: Record<string, string> = {
    large_refund: 'Large refund',
    safety_concern: 'Safety concern',
    delivery_privacy_concern: 'Delivery privacy',
  }
  const reasonColors: Record<string, string> = {
    large_refund: '#0f6e56',
    safety_concern: '#b91c1c',
    delivery_privacy_concern: '#b45309',
  }

  if (loading) return <p>Loading escalations...</p>

  const open = escalations.filter((e) => e.status === 'open')
  const resolved = escalations.filter((e) => e.status === 'resolved')

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: 18 }}>Open ({open.length})</h2>
        <button onClick={loadEscalations} style={{ padding: '4px 10px' }}>Refresh</button>
      </div>

      {open.length === 0 && <p style={{ color: '#888' }}>Nothing waiting on a human right now.</p>}

      {open.map((e) => (
        <div key={e.id} style={{ border: '1px solid #ccc', borderRadius: 8, padding: 12, marginBottom: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{
              background: reasonColors[e.reason] ?? '#666',
              color: 'white',
              borderRadius: 6,
              padding: '2px 8px',
              fontSize: 12,
            }}>
              {reasonLabels[e.reason] ?? e.reason}
            </span>
            <span style={{ fontSize: 12, color: '#888' }}>{new Date(e.created_at).toLocaleString()}</span>
          </div>
          {e.order_id && <div style={{ marginTop: 6, fontSize: 13 }}>Order #{e.order_id}</div>}
          <div style={{ marginTop: 6 }}>"{e.customer_message}"</div>
          <div style={{ marginTop: 6, fontSize: 13, color: '#555' }}>{e.agent_note}</div>
          <button onClick={() => resolve(e.id)} style={{ marginTop: 10, padding: '4px 10px' }}>
            Mark resolved
          </button>
        </div>
      ))}

      {resolved.length > 0 && (
        <>
          <h2 style={{ fontSize: 18, marginTop: 24 }}>Resolved ({resolved.length})</h2>
          {resolved.map((e) => (
            <div key={e.id} style={{ opacity: 0.5, padding: 8, fontSize: 13 }}>
              #{e.id} — {reasonLabels[e.reason] ?? e.reason} — "{e.customer_message}"
            </div>
          ))}
        </>
      )}
    </div>
  )
}

function App() {
  const [tab, setTab] = useState<'chat' | 'dashboard'>('chat')

  return (
    <div style={{ maxWidth: 600, margin: '40px auto', fontFamily: 'sans-serif' }}>
      <h1>cartwise</h1>

      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          onClick={() => setTab('chat')}
          style={{ padding: '6px 14px', fontWeight: tab === 'chat' ? 'bold' : 'normal' }}
        >
          Customer chat
        </button>
        <button
          onClick={() => setTab('dashboard')}
          style={{ padding: '6px 14px', fontWeight: tab === 'dashboard' ? 'bold' : 'normal' }}
        >
          Agent dashboard
        </button>
      </div>

      {tab === 'chat' ? <ChatView /> : <DashboardView />}
    </div>
  )
}

export default App