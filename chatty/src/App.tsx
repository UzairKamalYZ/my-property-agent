import { useState } from "react";
import "./App.css";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendPrompt = async () => {
    const prompt = input;
    const userMessage = { role: "user", content: prompt };
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setInput("");

    const eventSource = new EventSource(`/ask?prompt=${encodeURIComponent(prompt)}&stream=True`);
    let assistantMessage = { role: "assistant", content: "" };
    setMessages(prevMessages => [...prevMessages, assistantMessage]);

    eventSource.onmessage = (e) => {
      setMessages(prevMessages => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.role === "assistant") {
          const updatedLastMessage = { ...lastMessage, content: lastMessage.content + e.data };
          return [...prevMessages.slice(0, -1), updatedLastMessage];
        }
        return prevMessages;
      });
    };

    eventSource.onerror = (e) => {
      console.error("EventSource failed:", e);
      eventSource.close();
    };
  };

  return (
    <div className="chat-ui">
      <div className="messages">
        {messages.map((m, i) => <p key={i} data-role={m.role}>{m.content}</p>)}
      </div>
      <div className="input-area">
        <input value={input} onChange={e => setInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && sendPrompt()}/>
        <button onClick={sendPrompt}>Send</button>
      </div>
    </div>
  );
}
