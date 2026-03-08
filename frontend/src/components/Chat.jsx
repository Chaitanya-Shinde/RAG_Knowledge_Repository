import Message from "./Message";
import Input from "./Input";

export default function Chat({messages, setMessages}) {

  return (
    <div className="flex flex-col h-full">

      <div className="flex-1 overflow-y-auto p-6 space-y-4">

        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-32">
            Ask something about your documents
          </div>
        )}

        {messages.map((m, i) =>
          <Message key={i} message={m}/>
        )}

      </div>

      <Input messages={messages} setMessages={setMessages}/>

    </div>
  );
}