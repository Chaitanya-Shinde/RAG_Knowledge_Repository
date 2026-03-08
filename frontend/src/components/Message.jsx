export default function Message({message}) {

  if(message.role==="user"){
    return(
      <div className="text-right">
        <div className="inline-block bg-blue-500 text-white px-4 py-2 rounded-lg">
          {message.text}
        </div>
      </div>
    )
  }

  return(

    <div className="text-left">

      <div className="bg-gray-100 p-4 rounded-lg">

        <div className="text-sm whitespace-pre-wrap">
          {message.text}
        </div>

        {message.sources && (

          <div className="mt-3 text-xs text-gray-500">

            Sources:

            {message.sources.map((s,i)=>(
              <div key={i}>📄 {s.filename}</div>
            ))}

          </div>

        )}

      </div>

    </div>

  )
}