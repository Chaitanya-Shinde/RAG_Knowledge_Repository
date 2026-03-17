import { useState, useEffect } from "react";
import { getCurrentUser } from "./api";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";

export default function App() {

  const [messages, setMessages] = useState([]);

  useEffect(() => {

    async function checkAuth() {

      const user = await getCurrentUser();

      if (!user) {
        window.location.href = "http://localhost:8000/auth/login";
      }

    }

    checkAuth();

  }, []);

  return (
    <div className="h-screen flex bg-gray-50">

      {/* <div className=" flex flex-col">
        <h1>Hi </h1>
      </div> */}
        <Sidebar />
      
      <div className="flex-1 flex flex-col">
        <Chat messages={messages} setMessages={setMessages}/>
      </div>

    </div>
  );
}