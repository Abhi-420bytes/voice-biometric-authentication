import { Routes, Route } from 'react-router-dom'
import Navbar       from './components/Navbar'
import Home         from './pages/Home'
import Enroll       from './pages/Enroll'
import Authenticate from './pages/Authenticate'
import SpoofDetect  from './pages/SpoofDetect'
import Users        from './pages/Users'

export default function App() {
  return (
    <div className="min-h-screen bg-bg text-text font-mono">
      <Navbar />
      <Routes>
        <Route path="/"             element={<Home />} />
        <Route path="/enroll"       element={<Enroll />} />
        <Route path="/authenticate" element={<Authenticate />} />
        <Route path="/spoof"        element={<SpoofDetect />} />
        <Route path="/users"        element={<Users />} />
      </Routes>
    </div>
  )
}
