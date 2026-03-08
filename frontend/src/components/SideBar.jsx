import { useEffect, useState } from "react";
import { getDocuments, uploadFile, ingestDocs } from "../api";

export default function Sidebar() {

  const [docs, setDocs] = useState([]);

  async function loadDocs() {
    try {
      const data = await getDocuments();
      setDocs(data.documents || []);
    } catch (err) {
      console.error("Failed to load docs", err);
    }
  }

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadDocs();
  }, []);

  async function handleUpload(e) {

    const file = e.target.files[0];
    if (!file) return;

    alert("Uploading...");
    await uploadFile(file);

    alert("Ingesting...");
    await ingestDocs();

    await loadDocs();

    alert("Done!");
  }

  return (
    <div className="w-64 bg-white border-r p-4 hidden md:block">

      <h2 className="font-semibold mb-4">Documents</h2>

      <div className="space-y-2">

        {docs.map((d,i)=>(
          <div
            key={i}
            className="p-2 bg-gray-100 rounded text-sm"
          >
            📄 {d.filename}

            <div className="text-xs text-gray-500">
              {d.chunks} chunks
            </div>

          </div>
        ))}

      </div>

      <label className="block mt-4 bg-blue-500 text-white p-2 rounded text-center cursor-pointer">

        Upload

        <input
          type="file"
          className="hidden"
          onChange={handleUpload}
        />

      </label>

    </div>
  );
}