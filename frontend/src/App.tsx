import { useState } from 'react'

type Message = {
  role: 'user' | 'agent'
  text: string
}

type PendingAction = {
  action: string
  order_id: number
} | null

function App() {
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
    // Remember whether the agent is now waiting on a yes/no.
    setPendingAction(data.pending_action ?? null)
  }

  return (
    <div style={{ maxWidth: 500, margin: '40px auto', fontFamily: 'sans-serif' }}>
      <h1>cartwise support</h1>

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
            Waiting for your confirmation to cancel order #{pendingAction.order_id}...
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

export default App