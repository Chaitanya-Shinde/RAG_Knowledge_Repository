import { useState } from "react";
import { queryLLM } from "../api";

// eslint-disable-next-line no-unused-vars
export default function Input({messages , setMessages}) {

  const [text,setText] = useState("");
  const [loading,setLoading] = useState(false);
  const [model, setModel] = useState("gemini");

  async function send() {

    if(!text) return;

    const userMsg = {role:"user",text};

    setMessages(m=>[...m,userMsg]);

    setLoading(true);

    const res = await queryLLM(text,model);

    setMessages(m=>[
      ...m,
      {
        role:"bot",
        text:res.answer,
        sources:res.sources
      }
    ]);

    setText("");
    setLoading(false);
  }

  return (
    <div className="border-t p-4 flex gap-2">

      <input
        className="flex-1 border rounded px-3 py-2"
        value={text}
        onChange={e=>setText(e.target.value)}
        placeholder="Ask something..."
      />
      <select
        value={model}
        onChange={(e) => setModel(e.target.value)}
        className="border rounded px-2 py-1"
      >
        <option value="gemini">Gemini</option>
        <option value="ollama">Ollama (Llama3.2)</option>
      </select>

      <button
        onClick={send}
        className="bg-blue-500 text-white px-4 rounded"
      >
        Send
      </button>

      {loading && <div className="text-gray-400">Thinking...</div>}

    </div>
  );
}