import { X, Calculator } from 'lucide-react';

export function CalculationDashboard({ data, onClose }: { data: any, onClose: () => void }) {
  if (!data) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0, 0, 0, 0.4)',
      backdropFilter: 'blur(8px)',
      WebkitBackdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
      padding: '20px',
      overflowY: 'auto'
    }}>
      <div className="card" style={{
        width: '100%',
        maxWidth: '900px',
        maxHeight: '90vh',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(255, 255, 255, 0.85)',
        padding: '0',
        overflow: 'hidden'
      }}>
        
        {/* Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '20px 24px',
          borderBottom: '1px solid var(--border)',
          background: 'rgba(255, 255, 255, 0.5)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ padding: '8px', background: 'var(--accent-muted)', borderRadius: 'var(--radius-md)' }}>
              <Calculator size={24} color="var(--accent)" />
            </div>
            <h2 style={{ color: 'var(--text-main)', fontSize: '20px', margin: 0 }}>Calculation Dashboard</h2>
          </div>
          <button onClick={onClose} className="btn" style={{ padding: '8px', borderRadius: '50%' }}>
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div style={{
          padding: '24px',
          overflowY: 'auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))',
          gap: '24px'
        }}>
          
          {/* Section: Input / Reconstruction */}
          <div className="card" style={{ padding: '20px', background: 'rgba(255,255,255,0.6)' }}>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '16px', letterSpacing: '0.05em' }}>
              Input & Reconstruction
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Preprocessing Version</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.input?.preprocessing_version || 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Reconstruction Error (MSE)</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.input?.reconstruction_error?.toFixed(6) || 'Unavailable'}</span>
              </div>
            </div>
          </div>

          {/* Section: Anomaly Calculations */}
          <div className="card" style={{ padding: '20px', background: 'rgba(255,255,255,0.6)', gridRow: 'span 2' }}>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '16px', letterSpacing: '0.05em' }}>
              Anomaly Distance Metrics
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>MSE / Pixel Error <br/><span style={{fontSize:'11px'}}>MSE = 1/N * Σ(x - x̂)²</span></span>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.mse?.toFixed(6) ?? 'Unavailable'}</div>
                  {data.anomaly?.normalized_components?.pixel !== undefined && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Score: {data.anomaly.normalized_components.pixel.toFixed(4)}</div>}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>MAE (Mean Abs Error) <br/><span style={{fontSize:'11px'}}>MAE = 1/N * Σ|x - x̂|</span></span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.mae?.toFixed(6) ?? 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>SSIM Index <br/><span style={{fontSize:'10px', letterSpacing:'-0.5px'}}>SSIM = (2μ_x μ_y + C_1)(2σ_xy + C_2) / ((μ_x² + μ_y² + C_1)(σ_x² + σ_y² + C_2))</span></span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.ssim?.toFixed(6) ?? 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Structural Error <br/><span style={{fontSize:'11px'}}>1 - SSIM</span></span>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.structural_error?.toFixed(6) ?? 'Unavailable'}</div>
                  {data.anomaly?.normalized_components?.structural !== undefined && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Score: {data.anomaly.normalized_components.structural.toFixed(4)}</div>}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Perceptual Distance <br/><span style={{fontSize:'11px'}}>||f(x) - f(x̂)||²</span></span>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.perceptual_distance?.toFixed(6) ?? 'Unavailable'}</div>
                  {data.anomaly?.normalized_components?.perceptual !== undefined && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Score: {data.anomaly.normalized_components.perceptual.toFixed(4)}</div>}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Latent Distance <br/><span style={{fontSize:'11px'}}>||z - ẑ||²</span></span>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.latent_distance?.toFixed(6) ?? 'Unavailable'}</div>
                  {data.anomaly?.normalized_components?.latent !== undefined && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Score: {data.anomaly.normalized_components.latent.toFixed(4)}</div>}
                </div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Memory Distance <br/><span style={{fontSize:'11px'}}>Dist(z, M)</span></span>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.anomaly?.memory_distance?.toFixed(6) ?? 'Unavailable'}</div>
                  {data.anomaly?.normalized_components?.memory !== undefined && <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Calibrated Score: {data.anomaly.normalized_components.memory.toFixed(4)}</div>}
                </div>
              </div>
              
              <div style={{ borderTop: '1px solid var(--border)', margin: '8px 0' }}></div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent)', fontWeight: 600, fontSize: '16px' }}>
                <span>Final Hybrid Score <br/><span style={{fontSize:'12px', fontWeight:400}}>Σ (wᵢ x NormScoreᵢ)</span></span>
                <span style={{ fontFamily: 'monospace' }}>{data.anomaly?.hybrid_score?.toFixed(6) ?? 'Unavailable'}</span>
              </div>
              
              <div style={{ padding: '12px', background: 'rgba(0,0,0,0.03)', borderRadius: '8px', fontSize: '12px', fontFamily: 'monospace', color: 'var(--text-muted)', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
                Weights: {JSON.stringify(data.anomaly?.weights || {}).replace(/,/g, ', ')}
              </div>
            </div>
          </div>

          {/* Section: OOD */}
          <div className="card" style={{ padding: '20px', background: 'rgba(255,255,255,0.6)' }}>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '16px', letterSpacing: '0.05em' }}>
              OOD / Domain Shift
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>OOD Index <br/><span style={{fontSize:'11px'}}>Distance / Threshold</span></span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.ood?.score != null ? `${data.ood.score.toFixed(4)}x` : 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>OOD Threshold Index</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.ood?.threshold?.toFixed(4) ?? 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '8px' }}>
                <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>Decision</span>
                <span style={{ 
                  fontSize: '12px', fontWeight: 600, padding: '4px 8px', borderRadius: '4px',
                  background: data.ood?.decision?.includes('OUT') ? 'var(--error)' : 'var(--success)',
                  color: 'white'
                }}>
                  {data.ood?.decision || 'UNAVAILABLE'}
                </span>
              </div>
            </div>
          </div>

          {/* Section: Uncertainty & Calibration */}
          <div className="card" style={{ padding: '20px', background: 'rgba(255,255,255,0.6)' }}>
            <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '16px', letterSpacing: '0.05em' }}>
              Uncertainty & Calibration
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>TTA Runs</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.uncertainty?.runs ?? 'Unavailable'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>TTA Individual Scores</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600, fontSize: '11px' }}>
                    {data.uncertainty?.individual_scores ? data.uncertainty.individual_scores.map((s: number) => s.toFixed(6)).join(', ') : 'Unavailable'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>TTA Mean <br/><span style={{fontSize:'11px'}}>mean(scores)</span></span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.uncertainty?.mean?.toFixed(6) ?? 'Unavailable'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>TTA Variance <br/><span style={{fontSize:'11px'}}>population var(scores)</span></span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.uncertainty?.variance?.toFixed(6) ?? 'Unavailable'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Classification</span>
                  <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.uncertainty?.classification ?? 'Unavailable'}</span>
                </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Calibration Dataset</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.calibration?.dataset || 'Unavailable'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Anomaly Threshold</span>
                <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{data.calibration?.anomaly_threshold?.toFixed(4) ?? 'Unavailable'}</span>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
