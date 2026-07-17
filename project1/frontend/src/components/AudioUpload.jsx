import { useRef, useState } from 'react'

export default function AudioUpload({ label, multiple = false, onChange }) {
  const inputRef             = useRef()
  const mediaRef             = useRef()
  const chunksRef            = useRef([])
  const [files, setFiles]    = useState([])
  const [recording, setRec]  = useState(false)
  const [recFile, setRecFile] = useState(null)

  function handleFiles(selected) {
    const arr = Array.from(selected)
    setFiles(arr)
    onChange(multiple ? arr : arr[0])
  }

  async function startRecording() {
    chunksRef.current = []
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const rec    = new MediaRecorder(stream)
    mediaRef.current = rec
    rec.ondataavailable = e => chunksRef.current.push(e.data)
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop())
      const blob = new Blob(chunksRef.current, { type: 'audio/wav' })
      const file = new File([blob], 'recording.wav', { type: 'audio/wav' })
      setRecFile(file)
      onChange(multiple ? [file] : file)
    }
    rec.start()
    setRec(true)
  }

  function stopRecording() {
    mediaRef.current?.stop()
    setRec(false)
  }

  return (
    <div className="border border-border rounded-lg p-4 bg-bg">
      <p className="text-xs text-muted mb-3">{label}</p>

      {/* File upload */}
      <div
        className="border-2 border-dashed border-border rounded-lg p-5 text-center cursor-pointer hover:border-accent transition-colors mb-3"
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); handleFiles(e.dataTransfer.files) }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,audio/*"
          multiple={multiple}
          className="hidden"
          onChange={e => handleFiles(e.target.files)}
        />
        {files.length > 0
          ? <p className="text-green text-xs">{files.map(f => f.name).join(', ')}</p>
          : <p className="text-muted text-xs">Drop WAV file{multiple ? 's' : ''} here or click to browse</p>
        }
      </div>

      {/* Mic recording */}
      <div className="flex items-center gap-3">
        {!recording ? (
          <button
            onClick={startRecording}
            className="flex items-center gap-2 px-3 py-1.5 bg-red text-white text-xs rounded-md hover:opacity-80 transition"
          >
            <span className="w-2 h-2 rounded-full bg-white" />
            Record from mic
          </button>
        ) : (
          <button
            onClick={stopRecording}
            className="flex items-center gap-2 px-3 py-1.5 bg-border text-red text-xs rounded-md hover:opacity-80 transition"
          >
            <span className="w-2 h-2 rounded-full bg-red recording-ring inline-block" />
            Stop recording
          </button>
        )}
        {recFile && !recording && (
          <span className="text-green text-xs">✓ {recFile.name}</span>
        )}
      </div>
    </div>
  )
}
