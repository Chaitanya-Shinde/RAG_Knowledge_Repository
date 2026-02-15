const API_BASE = "http://127.0.0.1:8000";

async function send(){
  const p = document.getElementById('prompt').value;
  if(!p) return;
  append('user', p);
  const form = new FormData();
  form.append('prompt', p);
  form.append('k', 5);

  const res = await fetch(API_BASE + '/query', { method: 'POST', body: form });
  if(!res.ok){ append('bot', 'Error: '+res.statusText); return; }
  const data = await res.json();
  append('bot', data.answer + '\n\nSources: ' + (data.sources||[]).join(', '));
}


function append(cls, text){
  const div = document.getElementById('chat');
  const el = document.createElement('div');
  el.className = cls==='user'?'user':'bot';
  el.textContent = text;
  div.appendChild(el);
  div.scrollTop = div.scrollHeight;
}
