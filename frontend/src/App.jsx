import { useState, useEffect } from "react"
import axios from "axios"

// Full feature sets required by the backend models
const BREAST_FEATURES = [
  'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
  'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
  'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
  'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
  'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
  'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
];

const LUNG_FEATURES = [
  'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
  'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
  'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
  'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
  'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV',
  'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx'
];

export default function CancerDetectionUI() {
  const [cancerType, setCancerType] = useState("breast")
  const [features, setFeatures] = useState(Array(30).fill(0))
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const activeFields = cancerType === "breast" ? BREAST_FEATURES : LUNG_FEATURES

  useEffect(() => {
    // Reset features when type changes
    setFeatures(Array(30).fill(0))
    setResult(null)
    setError(null)
  }, [cancerType])

  const handleChange = (index, value) => {
    const updated = [...features]
    updated[index] = value === "" ? 0 : parseFloat(value)
    setFeatures(updated)
  }

  const handlePredict = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await axios.post("http://127.0.0.1:8000/predict", {
        cancer_type: cancerType,
        features: features
      })

      setResult(response.data)
    } catch (err) {
      console.error(err)
      setError("Prediction failed. Please ensure the backend server is running at http://127.0.0.1:8000")
    } finally {
      setLoading(false)
    }
  }

  const fillSample = () => {
    const sample = Array(30).fill(0).map(() => 
      cancerType === 'breast' ? (Math.random() * 20 + 5).toFixed(2) : Math.floor(Math.random() * 2)
    )
    setFeatures(sample.map(Number))
  }

  return (
    <div className="min-h-screen bg-black text-white overflow-x-hidden relative font-sans">
      {/* Animated Background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-0 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-0 w-[30rem] h-[30rem] bg-cyan-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 left-1/2 w-[25rem] h-[25rem] bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
      </div>

      {/* Grid Overlay */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none"></div>

      {/* Main Content */}
      <div className="relative z-10 px-6 py-10 max-w-7xl mx-auto">

        {/* Header */}
        <div className="text-center mb-14">
          <div className="inline-block px-4 py-2 rounded-full border border-pink-500/40 bg-pink-500/10 mb-6 backdrop-blur-xl">
            <span className="text-pink-300 tracking-widest text-xs font-bold uppercase">
              AI Powered Multi-Cancer Detection System
            </span>
          </div>

          <h1 className="text-6xl md:text-7xl font-black leading-tight bg-gradient-to-r from-pink-400 via-cyan-400 to-purple-400 text-transparent bg-clip-text animate-pulse">
            Cancer Detection
          </h1>

          <p className="mt-6 text-gray-400 text-lg max-w-3xl mx-auto leading-relaxed">
            Advanced Machine Learning platform predicting Breast and Lung cancer types using ensemble learning, real-time analytics, and intelligent risk assessment.
          </p>
        </div>

        {/* Main Grid */}
        <div className="grid lg:grid-cols-2 gap-8 items-start">

          {/* Left Card: Input Panel */}
          <div className="bg-white/10 backdrop-blur-2xl border border-white/10 rounded-3xl p-8 shadow-2xl hover:scale-[1.005] transition duration-500">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h2 className="text-3xl font-bold tracking-tight">Prediction Panel</h2>
                <p className="text-gray-400 mt-1">Real-time AI cancer analysis</p>
              </div>
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-pink-500 to-purple-500 flex items-center justify-center text-2xl shadow-lg">
                🧬
              </div>
            </div>

            {/* Cancer Type Selection */}
            <div className="mb-8">
              <label className="block mb-2 text-xs font-bold text-gray-400 uppercase tracking-widest">Select Cancer Type</label>
              <select 
                value={cancerType}
                onChange={(e) => setCancerType(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-2xl px-5 py-4 outline-none focus:border-pink-500 transition duration-300 cursor-pointer appearance-none"
              >
                <option value="breast">Breast Cancer (Wisconsin Dataset)</option>
                <option value="lung">Lung Cancer (Cervical Risk Factors)</option>
                <option value="skin" disabled>Skin Cancer (Coming Soon)</option>
              </select>
            </div>

            {/* Input Fields Grid */}
            <div className="flex justify-between items-center mb-4">
               <h3 className="text-sm font-bold text-gray-300 tracking-wider uppercase">Biological Features</h3>
               <button onClick={fillSample} className="text-[10px] text-cyan-400 font-bold border border-cyan-400/30 px-2 py-1 rounded hover:bg-cyan-400/10 transition">FILL SAMPLE DATA</button>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8 max-h-[350px] overflow-y-auto pr-4 custom-scrollbar">
              {activeFields.map((field, idx) => (
                <div key={field} className="space-y-1">
                  <label className="block text-[10px] font-bold text-gray-500 truncate uppercase tracking-tighter" title={field}>
                    {field}
                  </label>
                  <input
                    type="number"
                    value={features[idx] || ""}
                    onChange={(e) => handleChange(idx, e.target.value)}
                    placeholder="0.00"
                    className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 outline-none focus:border-cyan-400 transition text-sm font-mono"
                  />
                </div>
              ))}
            </div>

            {/* Predict Button */}
            <button 
              onClick={handlePredict}
              disabled={loading}
              className={`w-full py-5 rounded-2xl font-black text-xl tracking-[0.2em] uppercase shadow-xl transition duration-300 hover:scale-[1.02] active:scale-[0.98] ${
                loading 
                ? 'bg-gray-800 text-gray-600'
                : 'bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500 hover:shadow-pink-500/40'
              }`}
            >
              {loading ? 'Analyzing...' : 'Predict Cancer Risk'}
            </button>

            {/* Mini Stats */}
            <div className="grid grid-cols-3 gap-4 mt-8">
              <div className="bg-black/30 border border-white/10 rounded-2xl p-4 text-center">
                <h3 className="text-2xl font-bold text-pink-400">96%</h3>
                <p className="text-gray-500 text-[10px] uppercase font-bold mt-1">Accuracy</p>
              </div>
              <div className="bg-black/30 border border-white/10 rounded-2xl p-4 text-center">
                <h3 className="text-2xl font-bold text-cyan-400">2</h3>
                <p className="text-gray-500 text-[10px] uppercase font-bold mt-1">Models</p>
              </div>
              <div className="bg-black/30 border border-white/10 rounded-2xl p-4 text-center">
                <h3 className="text-2xl font-bold text-purple-400">AI</h3>
                <p className="text-gray-500 text-[10px] uppercase font-bold mt-1">Ensemble</p>
              </div>
            </div>

            {error && <p className="mt-4 text-red-500 text-xs text-center font-bold">{error}</p>}
          </div>

          {/* Right Panel: Result & Analytics */}
          <div className="space-y-8">

            {/* Result Card */}
            <div className={`transition-all duration-500 bg-gradient-to-br from-pink-500/20 to-purple-500/20 border border-pink-500/20 backdrop-blur-2xl rounded-3xl p-8 shadow-2xl relative overflow-hidden ${result ? 'opacity-100' : 'opacity-40'}`}>
              <div className="absolute top-0 right-0 w-40 h-40 bg-pink-500/20 rounded-full blur-3xl"></div>

              <div className="relative z-10">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-pink-300 uppercase tracking-widest text-[10px] font-bold">
                      Prediction Analysis
                    </p>
                    <h2 className={`text-5xl font-black mt-3 tracking-tighter ${result?.result === 'Malignant' || result?.result === 'Cancer' ? 'text-pink-500' : 'text-emerald-400'}`}>
                      {result ? result.result : 'Pending'}
                    </h2>
                  </div>
                  <div className={`text-7xl ${result ? 'animate-bounce' : ''}`}>
                    {result?.result === 'Malignant' || result?.result === 'Cancer' ? '⚠️' : '🩺'}
                  </div>
                </div>

                <div className="mt-10">
                  <div className="flex justify-between mb-3 text-gray-300 font-bold text-xs uppercase tracking-widest">
                    <span>Confidence Score</span>
                    <span className="text-white font-mono">{result ? (result.probability * 100).toFixed(1) : '00.0'}%</span>
                  </div>

                  <div className="w-full bg-black/40 rounded-full h-4 overflow-hidden p-1 border border-white/5">
                    <div 
                      className="h-full bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-400 rounded-full transition-all duration-1000 ease-out shadow-[0_0_15px_rgba(236,72,153,0.5)]"
                      style={{ width: `${result ? result.probability * 100 : 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Analytics Cards */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl hover:border-pink-500/30 transition">
                <div className="text-4xl mb-4">📊</div>
                <h3 className="text-xl font-bold mb-2 tracking-tight">ML Analytics</h3>
                <p className="text-gray-500 text-xs leading-relaxed">
                  Real-time machine learning evaluation with ensemble predictions and probabilistic risk profiling.
                </p>
              </div>

              <div className="bg-white/5 border border-white/10 rounded-3xl p-6 backdrop-blur-xl hover:border-cyan-500/30 transition">
                <div className="text-4xl mb-4">⚡</div>
                <h3 className="text-xl font-bold mb-2 tracking-tight">API Inference</h3>
                <p className="text-gray-500 text-xs leading-relaxed">
                  High-speed prediction pipeline served via optimized FastAPI endpoint for instant results.
                </p>
              </div>
            </div>

            {/* Workflow */}
            <div className="bg-black/40 border border-white/10 rounded-3xl p-8 backdrop-blur-xl">
              <h3 className="text-2xl font-bold mb-8 tracking-tighter">AI Processing Pipeline</h3>
              <div className="space-y-6">
                {[
                  { id: 1, color: 'bg-pink-500', title: 'Feature Validation', desc: 'Patient biological markers are verified and standardized.' },
                  { id: 2, color: 'bg-cyan-500', title: 'Neural Compute', desc: 'Ensemble models analyze multivariate feature vectors.' },
                  { id: 3, color: 'bg-purple-500', title: 'Diagnostic Output', desc: 'Classification and certainty index generated in real-time.' }
                ].map((step) => (
                  <div key={step.id} className="flex items-center gap-4 group">
                    <div className={`w-12 h-12 rounded-xl ${step.color} flex-shrink-0 flex items-center justify-center font-black text-xl group-hover:scale-110 transition shadow-lg shadow-black/50`}>
                      {step.id}
                    </div>
                    <div>
                      <h4 className="font-bold text-sm text-gray-200">{step.title}</h4>
                      <p className="text-gray-500 text-[10px] leading-snug">{step.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-16 text-center text-gray-600 text-[10px] font-bold uppercase tracking-[0.3em]">
          AI-Based Multi-Cancer Detection System • Advanced Neural Intelligence • 2026
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(255,255,255,0.02);
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(236,72,153,0.3);
        }
      `}</style>
    </div>
  )
}
