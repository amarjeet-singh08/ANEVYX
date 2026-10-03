import { UploadCloud } from 'lucide-react';
import { useState, useRef } from 'react';
import { ScanViewer } from './components/ScanViewer';
import { Sidebar } from './components/Sidebar';
import { CalculationDashboard } from './components/CalculationDashboard';

export default function App() {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [currentTab, setCurrentTab] = useState('analyze');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDashboard, setShowDashboard] = useState(false);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    setOriginalSrc(objectUrl);
    setResult(null);
    setErrorMsg(null);
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/v1/analyze', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to connect to backend. Please ensure the server is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderContent = () => {
    switch (currentTab) {
      case 'analyze':
      default:
        return (
          <>
            <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '32px' }}>
              <div>
                <h1 style={{ fontSize: '28px', color: 'var(--accent)' }}>Analyze Scan</h1>
                <p style={{ color: 'var(--text-muted)', marginTop: '8px' }}>
                  Upload a Chest X-Ray to evaluate normative deviations.
                </p>
              </div>
              
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }}
                accept="image/png, image/jpeg, .dcm"
              />
              <button 
                className="btn btn-primary" 
                style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px 24px', fontSize: '16px' }}
                onClick={() => fileInputRef.current?.click()}
                disabled={isAnalyzing}
              >
                <UploadCloud size={20} />
                {isAnalyzing ? 'Analyzing...' : 'Upload Scan'}
              </button>
            </header>

            <div className="content-wrapper" style={{ display: 'flex', gap: '24px', flex: 1, alignItems: 'flex-start', minHeight: 'auto' }}>
              {/* Main Viewer Area */}
              <div className="card viewers-container" style={{ flex: 1.5, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '16px', overflow: 'visible', background: 'rgba(255,255,255,0.4)' }}>
                 <ScanViewer 
                   title="Original Scan" 
                   imageSrc={result?.pipeline?.processed_b64 || originalSrc} 
                 />
                 <ScanViewer 
                   title="Normative Reconstruction" 
                   imageSrc={result?.inference?.reconstruction?.map || originalSrc}
                   
                 />
                 <ScanViewer 
                   title="Anomaly Overlay" 
                   imageSrc={result?.pipeline?.processed_b64 || originalSrc} 
                   heatmapSrc={result?.inference?.localization?.map || null}
                   
                 />
              </div>
              
              {/* Evidence Panel */}
              <div style={{ flex: 1.2, display: 'flex', flexDirection: 'column', gap: '24px', overflow: 'visible', minWidth: '350px' }}>
                <div className="card" style={{ display: 'flex', flexDirection: 'column', background: 'rgba(255,255,255,0.7)', minHeight: 'min-content' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--border)', paddingBottom: '12px' }}>
                    <h3 style={{ color: 'var(--text-main)', fontSize: '18px', fontWeight: 600, margin: 0 }}>
                      Analysis Summary
                    </h3>
                    {result && result.inference && result.inference.calculations && (
                      <button 
                        onClick={() => setShowDashboard(true)}
                        className="btn btn-secondary"
                        style={{ fontSize: '12px', padding: '6px 12px' }}
                      >
                        View Calculations
                      </button>
                    )}
                  </div>
                  
                  {!result && !isAnalyzing && !errorMsg && (
                    <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
                      No analysis active. Upload a scan to begin.
                    </div>
                  )}

                  {errorMsg && (
                    <div style={{ color: 'var(--error)', fontSize: '14px', textAlign: 'center', marginTop: '20px', padding: '16px', background: 'rgba(255, 59, 48, 0.1)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(255, 59, 48, 0.2)' }}>
                      {errorMsg}
                    </div>
                  )}
                  
                  {isAnalyzing && (
                    <div style={{ color: 'var(--accent)', fontSize: '14px', textAlign: 'center', marginTop: '40px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
                      <div className="spinner" style={{ width: '24px', height: '24px', border: '3px solid var(--accent-muted)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
                      <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
                      Computing Mathematical Evidence...
                    </div>
                  )}
                  
                  {result && result.inference && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Model Status</div>
                        <div style={{ fontWeight: 600, color: result.inference.model_status === 'TRAINED AND LOADED' ? 'var(--success)' : 'var(--warning)' }}>
                          {result.inference.model_status || 'UNKNOWN'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Image quality</div>
                        <div style={{ fontWeight: 500, color: result.quality?.status === 'REJECTED' ? 'var(--error)' : 'inherit' }}>
                          {result.quality?.status || 'ACCEPTED'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Anomaly evidence</div>
                        <div style={{ fontWeight: 600, color: result.inference.anomaly?.level?.includes('UNAVAILABLE') || result.inference.anomaly?.level?.includes('ABSTAINED') || result.inference.anomaly?.level === 'UNCALIBRATED' ? 'var(--text-muted)' : (result.inference.anomaly?.level !== 'CONSISTENT WITH NORMALITY' ? 'var(--warning)' : 'var(--success)') }}>
                          {result.inference.anomaly?.level} 
                          {result.inference.anomaly?.score != null && (
                            <span style={{ fontWeight: 400, marginLeft: '4px' }}>(Score: {result.inference.anomaly.score.toFixed(3)})</span>
                          )}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Uncertainty</div>
                        <div style={{ fontWeight: 500, color: result.inference.uncertainty?.level === 'UNAVAILABLE' ? 'var(--text-muted)' : 'inherit' }}>
                          {result.inference.uncertainty?.level} {result.inference.uncertainty?.variance != null && `(Var: ${result.inference.uncertainty.variance.toFixed(4)})`}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Input distribution</div>
                        <div style={{ fontWeight: 500, color: result.inference.ood?.status === 'UNAVAILABLE' ? 'var(--text-muted)' : 'inherit' }}>
                          {result.inference.ood?.status}
                        </div>
                      </div>
                      
                      {result.inference.clinical_report && (
                        <div style={{ marginTop: '8px', padding: '16px', background: 'rgba(0, 113, 227, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(0, 113, 227, 0.1)' }}>
                           <div style={{ fontSize: '12px', color: 'var(--accent)', fontWeight: 600, marginBottom: '8px' }}>DECISION</div>
                           <div style={{ fontSize: '14px', color: 'var(--text-main)', lineHeight: 1.5 }}>
                             {result.inference.clinical_report}
                           </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        );
    }
  };

  return (
    <div className="layout">
      <Sidebar currentTab={currentTab} setCurrentTab={setCurrentTab} />
      <main className="main-content">
        {renderContent()}
      </main>
      
      {showDashboard && result?.inference?.calculations && (
        <CalculationDashboard 
          data={result.inference.calculations} 
          onClose={() => setShowDashboard(false)} 
        />
      )}
    </div>
  );
}
